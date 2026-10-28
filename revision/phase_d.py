"""Reproducible one-column IJIES figures from saved post-hoc tables.

Every canvas is 85 mm wide, with 10 pt type at its intended print size.
Figures do not combine panels or pool repeated images as independent samples.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from .common import CLASSES

WIDTH = 85/25.4
COLORS = ['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9']
ANALYSIS = 'post hoc; primary results unchanged'


def _setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.labelsize':10,'axes.titlesize':10,'xtick.labelsize':10,
        'ytick.labelsize':10,'legend.fontsize':10,'svg.hashsalt':'IJIES20267101',
        'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})


def _save(ctx, fig, name, data):
    out=ctx.out/'figures'
    out.mkdir(parents=True,exist_ok=True)
    frame=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
    frame.to_csv(out/(name+'_data.csv'),index=False,lineterminator='\n',float_format='%.15g')
    fig.savefig(out/(name+'.png'),dpi=600,metadata={'Software':'IJIES20267101 deterministic revision script'})
    fig.savefig(out/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None,'Creator':'IJIES20267101 revision'})
    fig.savefig(out/(name+'.svg'),metadata={'Date':None,'Creator':'IJIES20267101 revision'})
    plt.close(fig)


def _read(ctx,name):
    return pd.read_csv(ctx.track(ctx.out/'tables'/(name+'.csv')))


def _code(backbone,seed,fmt,rule):
    return ('M' if backbone=='mobilenetv2' else 'E')+str(seed)+('F' if fmt=='fp32' else 'I')+('G' if rule=='global' else 'P')


def workflow(ctx):
    steps=[
        ('Audit dataset images','Hash, pHash and label conflicts\nDeduplicate before splitting'),
        ('Freeze one split','Train / validation / calibration / test\nShared across seeds and formats'),
        ('Train and export','Train-only weights and INT8 inputs\nMobileNetV2 and EfficientNetB0'),
        ('Select on validation','Temperature and one global +\none classwise rule per case'),
        ('Freeze selected rules','Verify hashes before\ncalibration inference'),
        ('Calibrate fixed rules','24 rules, 96 CP bounds\nFailed rules deploy all-defer'),
        ('Evaluate on held-out test','Primary results from frozen rules\nAdditional analyses post hoc')]
    fig,ax=plt.subplots(figsize=(WIDTH,6.8))
    fig.subplots_adjust(left=.03,right=.97,bottom=.02,top=.98)
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    rows=[]
    for i,(title,detail) in enumerate(steps):
        y=.925-i*.14
        ax.add_patch(FancyBboxPatch((.015,y-.058),.97,.116,boxstyle='round,pad=0.005',
            linewidth=.8,edgecolor=COLORS[0],facecolor='#F5F9FC'))
        ax.text(.5,y+.031,title,ha='center',va='center',fontsize=10,fontweight='bold')
        ax.text(.5,y-.017,detail,ha='center',va='center',fontsize=10,linespacing=1.2)
        if i<len(steps)-1:
            ax.annotate('',xy=(.5,y-.079),xytext=(.5,y-.059),arrowprops={'arrowstyle':'->','color':'#555555','lw':1})
        rows.append({'stage':i+1,'title':title,'detail':detail,'analysis':ANALYSIS})
    _save(ctx,fig,'Fig1_workflow',rows)


def primary_bounds(ctx,cert):
    keys=['backbone','seed','format','rule']
    cases=cert[keys].drop_duplicates().reset_index(drop=True)
    cases['code']=[_code(*x) for x in cases.itertuples(index=False,name=None)]
    data=cert.merge(cases,on=keys,validate='many_to_one')
    ymap={code:i for i,code in enumerate(cases.code)}
    data['y_order']=data.code.map(ymap)
    fig,ax=plt.subplots(figsize=(WIDTH,6.0))
    fig.subplots_adjust(left=.275,right=.91,top=.985,bottom=.17)
    markers=['o','s','^','D']
    for c,name in enumerate(CLASSES):
        sub=data[data['class']==name]
        ax.scatter(sub.upper_bound,sub.y_order,s=23 if c<3 else 32,
                   marker=markers[c],color=COLORS[c],label=name.capitalize(),zorder=3)
    ax.axvline(.1,color='#333333',linestyle='--',linewidth=1)
    ax.set_yticks(range(len(cases)),cases.code)
    ax.invert_yaxis();ax.set_ylim(len(cases)-.4,-.6)
    ax.set_xlim(0,max(.20,float(data.upper_bound.max())+.008))
    ax.set_xticks([0,.05,.1,.15,.2],['0','.05','.10','.15','.20'])
    ax.set_xlabel('Calibration upper bound')
    ax.grid(axis='x',color='#DDDDDD',linewidth=.5,zorder=0)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,
               columnspacing=.8,handletextpad=.3,bbox_to_anchor=(.5,.005))
    _save(ctx,fig,'Fig2_all_candidate_bounds',data)


def certified_coverage(ctx,cert):
    rows=[]
    for key,part in cert[cert.status=='CERTIFIED'].groupby(['backbone','seed','format','rule'],sort=False):
        backbone,seed,fmt,rule=key
        rows.append({'case_id':len(rows)+1,'code':_code(*key),'backbone':backbone,
            'seed':seed,'format':fmt,'rule':rule,
            'n_test':int(part.n_test.sum()),'accepted':int(part.accepted_test.sum()),
            'coverage':float(part.accepted_test.sum()/part.n_test.sum()),
            'worst_class_automatic_error':float(part.automatic_error_test.max()),
            'analysis':ANALYSIS})
    data=pd.DataFrame(rows)
    fig,ax=plt.subplots(figsize=(WIDTH,4.0))
    fig.subplots_adjust(left=.22,right=.92,top=.97,bottom=.38)
    handles=[]
    for row in data.itertuples():
        color=COLORS[[17,29,43].index(row.seed)]
        marker='o' if row.format=='fp32' else '^'
        ax.scatter(row.coverage*100,row.worst_class_automatic_error*100,s=45,
            marker=marker,color=color,facecolor=color if row.rule=='proposed' else 'white',zorder=3)
        handles.append(Line2D([],[],linestyle='',marker=marker,color=color,
            markerfacecolor=color if row.rule=='proposed' else 'white',label=row.code))
    ax.set_xlim(66,80);ax.set_ylim(0,4.5)
    ax.set_yticks([0,1,2,3,4])
    ax.set_xticks([68,72,76,80])
    ax.set_xlabel('Automatic coverage (%)')
    ax.set_ylabel('Worst-class auto. error (%)')
    ax.grid(color='#DDDDDD',linewidth=.5,zorder=0)
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,
        columnspacing=.7,handletextpad=.2,bbox_to_anchor=(.5,.01))
    _save(ctx,fig,'Fig3_certified_coverage_error',data)


def _class_decision_plot(ctx,part,name,title):
    # The bottom margin holds the axis label and a three-row legend without overlap.
    fig,ax=plt.subplots(figsize=(WIDTH,4.0 if title else 3.6))
    fig.subplots_adjust(left=.32,right=.97,top=.87 if title else .97,bottom=.345 if title else .385)
    y=np.arange(4)
    correct=part.accepted_test.to_numpy()-part.errors_test.to_numpy()
    errors=part.errors_test.to_numpy()
    deferred=part.n_test.to_numpy()-part.accepted_test.to_numpy()
    ax.barh(y,correct,color=COLORS[0],label='Correct auto.')
    ax.barh(y,errors,left=correct,color=COLORS[4],label='Error auto.')
    ax.barh(y,deferred,left=correct+errors,color='#BBBBBB',label='Deferred')
    for i,n in enumerate(part.n_test):
        ax.text(int(n)+4,i,str(int(n)),ha='left',va='center',fontsize=10)
    ax.set_xlim(0,max(part.n_test)*1.13)
    ax.set_yticks(y,[x.capitalize() for x in CLASSES]);ax.invert_yaxis()
    ax.set_xticks([0,100,200,300]);ax.set_xlabel('Images per true class')
    if title:ax.set_title(title,pad=8)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y',length=0)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',frameon=False,
        ncol=1,bbox_to_anchor=(.5,.005),labelspacing=.25)
    data=part.copy();data['correct_automatic']=correct;data['deferred']=deferred
    _save(ctx,fig,name,data)


def classwise_decisions(ctx,cert):
    all_rows=[]
    for key,part in cert[cert.status=='CERTIFIED'].groupby(['backbone','seed','format','rule'],sort=False):
        part=part.set_index('class').loc[CLASSES].reset_index()
        code=_code(*key)
        # The manuscript caption identifies Fig. 4; companions carry their case code.
        if key==('mobilenetv2',29,'int8','proposed'):
            name,title='Fig4_classwise_decisions',None
        else:
            name,title='Fig4_companion_'+code,code
        _class_decision_plot(ctx,part,name,title)
        for _,row in part.iterrows():
            all_rows.append({**row.to_dict(),'code':code,
                'correct_automatic':int(row.accepted_test-row.errors_test),
                'deferred':int(row.n_test-row.accepted_test)})
    pd.DataFrame(all_rows).to_csv(ctx.out/'figures/Fig4_all_certified_class_counts.csv',
                                index=False,lineterminator='\n',float_format='%.15g')


def calibration_power(ctx):
    power=_read(ctx,'C4_calibration_power')
    data=power[power.true_newcastle_risk.isin([.01,.02,.05])].copy()
    fig,ax=plt.subplots(figsize=(WIDTH,3.7))
    fig.subplots_adjust(left=.23,right=.93,top=.97,bottom=.37)
    handles=[]
    for i,risk in enumerate([.01,.02,.05]):
        for accounting,ls,short in [('primary_96_bounds','-','δ/96'),('IUT_24_rules','--','δ/24 IUT')]:
            sub=data[(data.true_newcastle_risk==risk)&(data.accounting==accounting)].sort_values('n_newcastle')
            line,=ax.plot(sub.n_newcastle,sub.probability_pass_Newcastle,color=COLORS[i],
                linestyle=ls,marker='o' if ls=='-' else 's',markersize=3,
                linewidth=1.3,label=f'r={risk*100:g}%, {short}')
            handles.append(line)
    ax.set_xlabel('Newcastle calibration images')
    ax.set_ylabel('P(Newcastle bound passes)')
    ax.set_xlim(75,510);ax.set_ylim(-.02,1.02)
    ax.set_xticks([84,200,350,500])
    ax.grid(color='#DDDDDD',linewidth=.5)
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,
        columnspacing=.7,handletextpad=.3,bbox_to_anchor=(.5,.005),labelspacing=.4)
    _save(ctx,fig,'Fig5_newcastle_calibration_power',data)


def confidence_curves(ctx):
    curves=_read(ctx,'C2_risk_coverage_curves')
    data=curves[(curves.backbone=='mobilenetv2')&(curves.seed==29)&(curves['format']=='fp32')].copy()
    labels={'msp':'MSP','msp_raw':'MSP','msp_temperature':'MSP + temp.',
            'temperature_msp':'MSP + temp.','margin':'Margin','entropy':'Neg. entropy',
            'negative_entropy':'Neg. entropy','doctor':'DOCTOR','energy':'Energy'}
    fig,ax=plt.subplots(figsize=(WIDTH,3.75))
    fig.subplots_adjust(left=.23,right=.92,top=.97,bottom=.35)
    for i,(score,part) in enumerate(data.groupby('score',sort=False)):
        part=part.sort_values('coverage')
        ax.plot(part.coverage*100,part.worst_class_automatic_error*100,
            color=COLORS[i],linewidth=1.3,label=labels.get(score,score),
            linestyle=['-','--','-.',':','-','--'][i])
    ax.axhline(10,color='#555555',linestyle=':',linewidth=1)
    ax.set_xlabel('Automatic coverage (%)')
    ax.set_ylabel('Worst-class auto. error (%)')
    ax.set_xlim(0,100);ax.set_ylim(bottom=0)
    ax.set_xticks([0,25,50,75,100])
    ax.grid(color='#DDDDDD',linewidth=.5)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,
        columnspacing=.7,handletextpad=.3,bbox_to_anchor=(.5,.005),labelspacing=.4)
    _save(ctx,fig,'Fig6_confidence_risk_coverage',data)


def run(ctx):
    _setup()
    cert=_read(ctx,'A1_candidate_class_certificates')
    print('Phase D: vector workflow and primary figures',flush=True)
    workflow(ctx)
    primary_bounds(ctx,cert)
    certified_coverage(ctx,cert)
    classwise_decisions(ctx,cert)
    print('Phase D: two post-hoc exploratory figures',flush=True)
    calibration_power(ctx)
    confidence_curves(ctx)
    ctx.write_json('figures/figure_specification.json',{
        'canvas_width_mm':85,'font_size_pt':10,'png_dpi':600,
        'formats':['PNG','PDF','SVG'],'analysis':ANALYSIS,
        'Fig1':'Vector redraw of original workflow; original image2 711x1848 at 85 mm is approximately 212 dpi.',
        'Fig2':'All 24 rules x 4 original CP bounds, unchanged primary accounting.',
        'Fig3':'All seven primary certified rules; descriptive test error, not new certificates.',
        'Fig4':'Main illustration fixed to MobileNetV2 seed29 INT8 proposed; six companions and full counts for all seven.',
        'Fig5':'Newcastle stratum only; analytic probability is not full-policy certification probability.',
        'Fig6':'MobileNetV2 seed29 FP32 illustration selected post-hoc; full curves all cases in C2 table.',
        'case_code':'M=MobileNetV2; E=EfficientNetB0; seed17/29/43; F=FP32/I=INT8; G=global/P=proposed.'})
