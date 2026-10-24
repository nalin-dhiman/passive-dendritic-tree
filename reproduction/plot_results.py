#!/usr/bin/env python3
"""Draw quantitative figures from a completed public reproduction output.

python reproduction/plot_results.py --results RESULTS --output RESULTS/figures
This script reads calculated arrays and tables; it never runs a simulation.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'passive-cable-plot-cache'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'passive-cable-plot-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

COLORS={'baseline':'#4c554d','uniform':'#346b8e','selective':'#b56337'}
LABELS={'baseline':'Original','uniform':'Uniform thinning','selective':'Selective thinning'}
CRITERIA=['root_gain','all_port_gain','response_relative']
PRIMARY=['Somatic gain discrepancy','Distributed gain discrepancy','Complete-response-relative discrepancy']
plt.rcParams.update({'text.usetex':False,'font.family':'serif','font.serif':['DejaVu Serif'],
 'mathtext.fontset':'cm','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
 'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.6,'grid.color':'#dedfd9','grid.linewidth':.5,
 'pdf.fonttype':42,'savefig.facecolor':'white'})


def read_csv(path):
    with Path(path).open(newline='') as handle:return list(csv.DictReader(handle))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1048576),b''):h.update(block)
    return h.hexdigest()


class Figures:
    def __init__(self,results,output):
        self.root=Path(results).resolve();self.output=Path(output).resolve()
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError('Refusing to overwrite nonempty figure output: '+str(self.output))
        self.output.mkdir(parents=True,exist_ok=True)
        self.y=self.root/'y_cable';self.a=self.root/'allen';self.records=[];self.skipped=[]
        self.config=json.loads((self.y/'config.json').read_text())
        self.names=[s['id'] for s in self.config['scope']['settings']]
        self.ygeometries={g['id']:np.asarray(g['radii']) for g in self.config['geometries']}

    def save(self,fig,name,files,scope):
        # All legends are placed beyond the plotting axes with reserved margins.
        pdf=self.output/(name+'.pdf');png=self.output/(name+'.png')
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer();frame=fig.bbox
        hidden_ticks=set()
        for ax in fig.axes:
            for axis in [ax.xaxis,ax.yaxis]:
                low,high=sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                    if not low<=tick.get_loc()<=high: hidden_ticks.update([id(tick.label1),id(tick.label2)])
        for text in fig.findobj(match=matplotlib.text.Text):
            if id(text) in hidden_ticks or not text.get_visible() or not text.get_text(): continue
            box=text.get_window_extent(renderer)
            if box.width and box.height and (box.x0<frame.x0-2 or box.y0<frame.y0-2 or box.x1>frame.x1+2 or box.y1>frame.y1+2):
                raise RuntimeError('Text outside figure canvas: '+name+': '+text.get_text())
        fig.savefig(pdf);fig.savefig(png,dpi=180);plt.close(fig)
        sources=[]
        for file in sorted(set(Path(f).absolute() for f in files)):
            sources.append({'path':str(file.relative_to(self.root)),'sha256':digest(file)})
        combined=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
        self.records.append({'figure_pdf':pdf.name,'figure_png':png.name,'data_files':json.dumps(sources,separators=(',',':')),
            'combined_source_sha256':combined,'scope':scope})

    def geometry_y(self):
        fig=plt.figure(figsize=(9.3,5.7));gs=fig.add_gridspec(2,3,height_ratios=[2.1,1.15],hspace=.45)
        xyz=np.array([[0,0],[0,400],[-320,640],[320,640],[0,200],[-160,520],[160,520]])
        edges=[(0,1,0),(1,2,1),(1,3,2)]
        for column,name in enumerate(['baseline','uniform','selective']):
            ax=fig.add_subplot(gs[0,column]);r=self.ygeometries[name]
            for start,end,region in edges:
                color=COLORS[name] if name!='selective' or region==1 else COLORS['baseline']
                ax.plot(xyz[[start,end],0],xyz[[start,end],1],color=color,lw=4*r[region],solid_capstyle='round')
                tx,ty=[(50,200),(-230,430),(230,430)][region]
                ax.text(tx,ty,f'{r[region]:.4g} '+r'$\mu$m',fontsize=8,
                    ha='left' if region==0 else 'center',color=color)
            for j,(x,y) in enumerate(xyz):
                ax.plot(x,y,'o',ms=3,mfc='white',mec='#3d423c',mew=.7)
                ax.annotate(f'P{j}',(x,y),xytext=(-9,-11) if j in [0,2,4,5] else (7,5),textcoords='offset points',fontsize=7)
            ax.set_title(LABELS[name]);ax.set_aspect('equal');ax.set_xlim(-410,410);ax.set_ylim(-70,720);ax.axis('off')
            ax.plot([150,250],[-25,-25],color='#3d423c',lw=1);ax.text(200,-62,r'100 $\mu$m',ha='center',fontsize=7)
        for column,(key,title) in enumerate([('volume_um3','Cylindrical volume'),('area_um2','Lateral membrane area'),('axial','Local axial resistance')]):
            ax=fig.add_subplot(gs[1,column]);values=[]
            base=self.ygeometries['baseline']
            for name in ['baseline','uniform','selective']:
                r=self.ygeometries[name]
                if key=='volume_um3':value=np.sum(r*r)/np.sum(base*base)
                elif key=='area_um2':value=np.sum(r)/np.sum(base)
                else:value=(base[1]/r[1])**2
                values.append(100*value)
            ax.bar(np.arange(3),values,color=[COLORS[n] for n in ['baseline','uniform','selective']],width=.6)
            ax.set_xticks(range(3),['Original','Uniform','Selective']);ax.set_ylabel('% of original');ax.set_title(title)
            ax.set_ylim(0,max(values)*1.2);ax.grid(axis='y',alpha=.6);ax.set_axisbelow(True)
            for j,v in enumerate(values):ax.text(j,v+max(values)*.025,f'{v:.2f}%',ha='center',fontsize=8)
            if key=='axial':ax.set_title('Selected daughter axial resistance')
        fig.subplots_adjust(left=.07,right=.98,bottom=.09,top=.92,wspace=.5)
        fig.suptitle('Matched 5.91% material reduction: different spatial allocation',fontsize=11)
        self.save(fig,'01_y_geometry_material',[self.y/'config.json'],
            'Physical cylinder lengths and radii; display widths enlarged. Materials are analytic cylinder ratios; daughter1 resistance scales as radius^-2.')

    def y_electrical(self):
        fig,axs=plt.subplots(1,3,figsize=(10.5,3.8));sources=[]
        fields=['gain_root_by_frequency','gain_ports_by_frequency','response_relative_by_frequency']
        titles=['Root row-normalized\ngain discrepancy','All-port row-normalized\ngain discrepancy','Complete-response-\nrelative discrepancy']
        for j,(field,title) in enumerate(zip(fields,titles)):
            ax=axs[j]
            for name in ['uniform','selective']:
                values=[]
                for setting in self.names:
                    path=self.y/'raw_data'/f'comparison_{name}_{setting}.npz';sources.append(path)
                    with np.load(path) as z:
                        allocation=z['gain_allocation_by_frequency' if j<2 else 'response_relative_allocation_by_frequency']
                        values.append(100*(z[field]+allocation));frequency=z['frequency_Hz']
                values=np.asarray(values);low,high=values.min(axis=0),values.max(axis=0)
                ax.fill_between(frequency,low,high,color=COLORS[name],alpha=.12)
                ax.plot(frequency,high,color=COLORS[name],lw=1.5,label=LABELS[name])
            ax.axhline(5,color='#777167',ls='--',lw=.8,label='5% comparison' if j==0 else None)
            ax.set_xlabel('Frequency (Hz)');ax.set_ylabel('Error with allowance (%)');ax.set_title(title)
            ax.set_ylim(bottom=0);ax.set_xlim(0,100);ax.grid(alpha=.5)
        handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,frameon=False)
        fig.subplots_adjust(left=.065,right=.985,bottom=.23,top=.79,wspace=.48)
        fig.suptitle('Upper curves: maxima over seven engineering parameter settings',fontsize=11)
        self.save(fig,'02_y_electrical_criteria',sources,
            'Seven-setting extrema of sampled metrics plus unchanged empirical discrepancy allocations; shading spans deterministic engineering contrasts, not biological confidence intervals.')

    def geometry_allen(self):
        path=self.a/'retained_geometry.npz';material=self.a/'material_integrals.csv'
        with np.load(path) as data:
            swc=data['swc'];parent=data['parent'];ports=data['ports'];scales={n:data['scale_'+n] for n in ['baseline','uniform','selective']}
        xyz=np.c_[swc[:,2]-swc[0,2],swc[0,3]-swc[:,3]]
        fig=plt.figure(figsize=(9.6,6.4));gs=fig.add_gridspec(2,3,height_ratios=[3,1.1],hspace=.3)
        ext=xyz.max(axis=0)-xyz.min(axis=0);margin=.12*ext
        for column,name in enumerate(['baseline','uniform','selective']):
            ax=fig.add_subplot(gs[0,column]);segments=np.stack((xyz[parent[1:]],xyz[1:]),axis=1)
            widths=np.clip(1.7*swc[1:,5]*scales[name][1:],.35,3)
            colors=[COLORS[name] if name!='selective' or scale<1 else '#93978d' for scale in scales[name][1:]]
            ax.add_collection(LineCollection(segments,linewidths=widths,colors=colors))
            ax.add_patch(Circle(xyz[0],swc[0,5],facecolor='#646b60',edgecolor='none'))
            offsets=[(9,-10),(-21,2),(-20,2),(-15,5),(13,10),(15,-14),(8,-3)]
            for j,index in enumerate(ports):
                ax.plot(*xyz[index],'o',ms=2.6,mfc='white',mec='#323b32',mew=.5)
                ax.annotate(f'P{j}',xyz[index],xytext=offsets[j],textcoords='offset points',fontsize=6.8,
                    arrowprops={'arrowstyle':'-','color':'#888d80','lw':.45})
            ax.set_xlim(xyz[:,0].min()-margin[0],xyz[:,0].max()+margin[0]);ax.set_ylim(xyz[:,1].min()-margin[1],xyz[:,1].max()+margin[1]);ax.set_aspect('equal');ax.axis('off');ax.set_title(LABELS[name])
            left=xyz[:,0].max()-100;bottom=xyz[:,1].min()-margin[1]/2
            ax.plot([left,left+100],[bottom,bottom],color='#555b50',lw=.8);ax.text(left+50,bottom-20,r'100 $\mu$m',ha='center',fontsize=7)
        ax=fig.add_subplot(gs[1,:]);rows=read_csv(material);selected={name:next(r for r in rows if r['candidate']==name and int(r['mesh'])==8) for name in ['baseline','uniform','selective']}
        keys=['dendritic_volume_um3','dendritic_area_um2','whole_modeled_volume_um3','whole_capacitance_pF'];labels=['Dendritic volume','Dendritic membrane area','Soma + dendrite volume','Membrane capacitance']
        positions=np.arange(len(keys));width=.23
        for j,name in enumerate(['baseline','uniform','selective']):
            values=[100*float(selected[name][key])/float(selected['baseline'][key]) for key in keys]
            ax.bar(positions+(j-1)*width,values,width,color=COLORS[name],label=LABELS[name])
            if name!='baseline':
                for x,v in zip(positions+(j-1)*width,values):ax.text(x,v+.25,f'{v:.2f}',ha='center',fontsize=7,rotation=90)
        ax.axhline(100,color='#888',ls=':',lw=.7);ax.set_ylabel('% of original');ax.set_ylim(85,106);ax.set_xticks(positions,labels);ax.grid(axis='y',alpha=.4);ax.set_axisbelow(True)
        fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,frameon=False)
        fig.subplots_adjust(left=.07,right=.985,top=.91,bottom=.13,wspace=.25)
        fig.suptitle('Retained cortical arbor: fixed soma and prescribed dendritic thinning',fontsize=11)
        self.save(fig,'03_allen_geometry_material',[path,material,self.a/'geometry_and_ports.json'],
            'Actual retained2534node XY centerlines, vertically flipped for display. Widths enlarged; soma fixed. Ratios distinguish dendritic and soma-plus-dendrite material.')

    def allen_electrical(self):
        path=self.a/'electrical_equivalence.csv';rows=read_csv(path)
        fig,axs=plt.subplots(1,3,figsize=(10.5,3.8))
        for j,criterion in enumerate(CRITERIA):
            ax=axs[j]
            for name in ['uniform','selective']:
                data=sorted([r for r in rows if r['candidate']==name and r['criterion']==criterion],key=lambda r:float(r['frequency_Hz']))
                x=np.log10(1+np.array([float(r['frequency_Hz']) for r in data]));y=100*np.array([float(r['error_plus_allowance']) for r in data])
                ax.plot(x,y,'o-',ms=3,lw=1.4,color=COLORS[name],label=LABELS[name])
            ax.axhline(5,color='#777167',ls='--',lw=.8,label='5% comparison' if j==0 else None)
            ticks=[0,1,2,5,10,20,50,100];ax.set_xticks(np.log10(1+np.array(ticks)),[str(v) for v in ticks]);ax.set_xlabel('Frequency (Hz; log(1 + f) spacing)');ax.set_ylabel('Error with allowance (%)');ax.set_title(['Somatic gain\ndiscrepancy','Distributed gain\ndiscrepancy','Complete-response-\nrelative discrepancy'][j]);ax.grid(alpha=.4);ax.set_ylim(bottom=0)
        fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,frameon=False)
        fig.subplots_adjust(left=.065,right=.985,bottom=.24,top=.79,wspace=.48)
        fig.suptitle('Primary normalization: somatic and distributed gains use the full baseline norm',fontsize=11)
        self.save(fig,'04_allen_primary_criteria',[path],
            'Primary root-row and distributed spectral gains normalized by full baseline impedance spectral norm; complete response uses D inverse(Z0). Eight sampled frequencies and unchanged numerical allowances.')

    def harmonic_allen(self):
        path=self.a/'impedance_matrices.npz'
        with np.load(path) as data:
            frequencies=data['frequencies_Hz'];q=list(frequencies).index(10);z={n:data[n+'_k8'][q] for n in ['baseline','uniform','selective']}
        times=np.linspace(0,200,801);phase=np.exp(2j*np.pi*10*times/1000);fig,axs=plt.subplots(2,2,figsize=(9,5.4))
        for row,source in enumerate([3,4]):
            for col,observe in enumerate([0,source]):
                ax=axs[row,col]
                for name in ['baseline','uniform','selective']:
                    ax.plot(times,np.real(.001*z[name][observe,source]*phase),color=COLORS[name],lw=1.3,label=LABELS[name])
                du=abs(.001*(z['uniform'][observe,source]-z['baseline'][observe,source]));ds=abs(.001*(z['selective'][observe,source]-z['baseline'][observe,source]))
                ax.set_title(f'Input P{source}; '+('somatic P0' if observe==0 else f'local P{observe}')+f' response\nDifference amplitudes (mV): U {du:.3g}, S {ds:.3g}',fontsize=8.8)
                ax.set_xlabel('Time within periodic steady state (ms)');ax.set_ylabel('Voltage relative to rest (mV)');ax.grid(alpha=.4);ax.set_xlim(0,200)
        handles,_=axs[0,0].get_legend_handles_labels();fig.legend(handles,['Original','Uniform thinning (U)','Selective thinning (S)'],loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,frameon=False)
        fig.subplots_adjust(left=.09,right=.985,bottom=.16,top=.86,hspace=.78,wspace=.35)
        fig.suptitle('10 Hz, 1 pA harmonic input: both current location and observation matter',fontsize=11)
        self.save(fig,'05_allen_harmonic_voltages',[path],
            'Fresh finest-mesh impedance phasors at10Hz, single1pA inputs atP3/P4. Two cycles of relative voltage; no temporal integration or pulse-startup claim.')

    def witnesses(self):
        yfiles=[self.y/'raw_data'/f'witness_{name}_nominal_root_null.npz' for name in ['uniform','selective']];afile=self.a/'root_null_witness.npz'
        examples=[]
        for name,path in zip(['uniform','selective'],yfiles):
            with np.load(path) as d:examples.append((f'Y: {LABELS[name]}',float(d['frequency_Hz']),d['input_phasor_nA']*1000,d['Vdelta_phasor_mV']))
        with np.load(afile) as d:examples.append(('Cortical arbor: selective thinning',float(d['frequency_Hz']),d['current'],d['difference']*.001))
        fig,axs=plt.subplots(3,2,figsize=(9.2,7.1));ports=np.arange(7)
        for row,(title,f,current,delta) in enumerate(examples):
            ax=axs[row,0];ax.bar(ports-.18,current.real,.36,color='#346b8e',label='Real current');ax.bar(ports+.18,current.imag,.36,color='#b56337',label='Imaginary current')
            ax.set_title(f'{title}; {f:g} Hz\nCurrent L2 norm {np.linalg.norm(current):g} pA',fontsize=9);ax.set_xticks(ports,[f'P{j}' for j in ports]);ax.set_ylabel('Current phasor (pA)');ax.axhline(0,color='#666',lw=.7);ax.grid(axis='y',alpha=.35);ax.set_axisbelow(True)
            ax=axs[row,1];ax.bar(ports,abs(delta),.65,color=['#7a887b']+['#a85b39']*6);ax.set_xticks(ports,[f'P{j}' for j in ports]);ax.set_ylabel('Difference amplitude (mV)');ax.set_title(f'Root difference {abs(delta[0]):.2g} mV; local maximum {max(abs(delta)):.4g} mV',fontsize=8.8);ax.grid(axis='y',alpha=.35);ax.set_axisbelow(True)
        fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.005),ncol=2,frameon=False)
        fig.subplots_adjust(left=.09,right=.985,bottom=.10,top=.91,wspace=.32,hspace=.65)
        fig.suptitle('Root-difference-null constructions retain measurable local voltage changes',fontsize=11)
        self.save(fig,'06_root_null_currents',yfiles+[afile],
            'Algebraically constructed signed/phase-controlled7port currents; Y norm10pA, cortical norm1pA. Cortical stored unitcurrent solution scaled by.001. Not synaptic stimuli.')

    def numerical_validation(self):
        files=[];meshes=[16,48,144];errors=np.zeros(3);independent=0
        for name in self.ygeometries:
            for setting in self.names:
                p=self.y/'processed_data'/f'harmonic_{name}_{setting}.json';files.append(p);d=json.loads(p.read_text())['details']
                errors=np.maximum(errors,[d['FVM_relative_row_errors'][str(n)] for n in meshes]);independent=max(independent,d['continuum_reference_relative'])
        a_path=self.a/'spatial_convergence.csv';algebra_path=self.a/'algebraic_checks.csv';rows=read_csv(a_path);files+=[a_path,algebra_path]
        fine=[2,4,8];aerrors=[max(float(r['relative_operator_difference']) for r in rows if int(r['fine'])==k) for k in fine]
        algebra=max(float(r['independent_relative_error']) for r in read_csv(algebra_path))
        fig,axs=plt.subplots(1,2,figsize=(9,3.8));ax=axs[0];ax.loglog(meshes,errors,'o-',color=COLORS['uniform'],label='Largest FVM row discrepancy');ax.loglog(meshes,errors[0]*(np.array(meshes)/meshes[0])**-2,'--',color='#7c8b6e',label='Second-order reference');ax.axhline(.0005,color='#777167',ls=':',label='Finest-mesh requirement');ax.set_xticks(meshes,[str(x) for x in meshes]);ax.set_xlabel('Intervals per cylinder');ax.set_ylabel('Normalized row discrepancy');ax.set_title(f'Y: independent continuum agreement {independent:.2g}',fontsize=9);ax.grid(which='both',alpha=.3)
        ax=axs[1];ax.loglog(fine,aerrors,'o-',color=COLORS['selective'],label='Largest adjacent-mesh discrepancy');ax.loglog(fine,aerrors[0]*(np.array(fine)/fine[0])**-2,'--',color='#7c8b6e',label='Second-order reference');ax.axhline(.0025,color='#777167',ls=':',label='Finest-pair requirement');ax.set_xticks(fine,[str(x) for x in fine]);ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());ax.set_xlabel('Fine subdivisions per source edge');ax.set_ylabel('Relative impedance operator difference');ax.set_title(f'Cortical arbor: independent solvers agree within {algebra:.2g}',fontsize=9);ax.grid(which='both',alpha=.3)
        for ax in axs:ax.legend(loc='upper center',bbox_to_anchor=(.5,-.25),frameon=False,fontsize=7)
        fig.subplots_adjust(left=.09,right=.98,bottom=.35,top=.84,wspace=.35);fig.suptitle('Spatial refinement and independent harmonic formulations',fontsize=11)
        self.save(fig,'07_numerical_validation',files,
            'WorstYrow discrepancies across35conditions versus exactcontinuum; corticaladjacentmeshes acrosspresets/frequencies. Norms differ and are labeled separately. Reference slopes are guides, not fitted error bounds.')

    def temporal_failure(self):
        table=self.a/'temporal_convergence.csv'
        if not table.exists():
            self.skipped.append({'figure':'08_allen_transient_accuracy','reason':'Transient outputs absent; run the reproduction with --include-transients.'});return
        rows=read_csv(table);finest=[r for r in rows if float(r['dt_ms'])==.025];files=[table,self.a/'transient_gate.json'];fig,axs=plt.subplots(1,2,figsize=(9.6,3.6))
        for name in ['baseline','uniform','selective']:
            p0=self.a/f'transient_{name}_P3_dt0.05.npz';p1=self.a/f'transient_{name}_P3_dt0.025.npz';files +=[p0,p1]
            with np.load(p0) as coarse,np.load(p1) as fine:
                times=fine['time_ms'];error=np.max(abs(fine['voltage_deviation_mV']-coarse['voltage_deviation_mV']),axis=1)
            axs[0].plot(times,error,color=COLORS[name],lw=1.2,label=LABELS[name])
        axs[0].axhline(.0025,color='#777167',ls='--',lw=.8);axs[0].set_yscale('symlog',linthresh=1e-5);axs[0].set_ylim(0,.12);axs[0].set_xlim(0,250);axs[0].set_xlabel('Time (ms)');axs[0].set_ylabel('Maximum port voltage difference (mV)');axs[0].set_title('Distal input: 0.025 versus 0.05 ms step');axs[0].grid(alpha=.35)
        groups=['baseline','uniform','selective'];x=np.arange(3);width=.32
        for j,(port,color) in enumerate([(0,'#346b8e'),(3,'#b56337')]):
            values=[float(next(r for r in finest if r['candidate']==name and int(r['input_port'])==port)['max_common_time_difference_mV']) for name in groups]
            axs[1].bar(x+(j-.5)*width,values,width,color=color,label=f'Input P{port}')
            for xx,v in zip(x+(j-.5)*width,values):axs[1].text(xx,v*1.12,f'{v:.3g}',ha='center',fontsize=7)
        axs[1].axhline(.0025,color='#777167',ls='--',lw=.8,label='0.0025 mV criterion');axs[1].set_yscale('log');axs[1].set_xticks(x,['Original','Uniform','Selective']);axs[1].set_ylim(1e-5,.15);axs[1].set_ylabel('Finest-pair maximum difference (mV)');axs[1].set_title('Required temporal accuracy remains failed');axs[1].grid(axis='y',alpha=.35);axs[1].set_axisbelow(True)
        for ax in axs:ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3,frameon=False,fontsize=7)
        fig.subplots_adjust(left=.08,right=.98,bottom=.31,top=.84,wspace=.35);fig.suptitle('Retained rectangular-current temporal accuracy result',fontsize=11)
        self.save(fig,'08_allen_transient_accuracy',files,
            'Nativecommon0.1ms recordedtimes; error over sevenports, dt.025vs.05ms; original.0025mV threshold unchanged. Reproducing this FAIL is not a transient accuracy PASS.')

    def finish(self):
        with (self.output/'FIGURE_PROVENANCE.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['figure_pdf','figure_png','data_files','combined_source_sha256','scope']);writer.writeheader();writer.writerows(self.records)
        (self.output/'figure_generation.json').write_text(json.dumps({'status':'PASS','figures':len(self.records),'formats':['PDF','PNG'],'skipped':self.skipped,'source_policy':'Calculated outputs only; no simulation or private input access.','matplotlib':matplotlib.__version__},indent=2)+'\n')
        print(json.dumps({'status':'PASS','figures':len(self.records),'skipped':self.skipped},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    figures=Figures(args.results,args.output)
    for method in [figures.geometry_y,figures.y_electrical,figures.geometry_allen,figures.allen_electrical,figures.harmonic_allen,figures.witnesses,figures.numerical_validation,figures.temporal_failure]:method()
    figures.finish()

if __name__=='__main__':main()
