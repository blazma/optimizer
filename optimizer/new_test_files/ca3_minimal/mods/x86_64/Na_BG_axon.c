/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__Na_BG_axon
#define _nrn_initial _nrn_initial__Na_BG_axon
#define nrn_cur _nrn_cur__Na_BG_axon
#define _nrn_current _nrn_current__Na_BG_axon
#define nrn_jacob _nrn_jacob__Na_BG_axon
#define nrn_state _nrn_state__Na_BG_axon
#define _net_receive _net_receive__Na_BG_axon 
#define _f_rates _f_rates__Na_BG_axon 
#define rates rates__Na_BG_axon 
#define states states__Na_BG_axon 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define gmax _p[0]
#define X_v0 _p[1]
#define X_k0 _p[2]
#define X_kt _p[3]
#define X_gamma _p[4]
#define X_tau0 _p[5]
#define Y_v0 _p[6]
#define Y_k0 _p[7]
#define Y_kt _p[8]
#define Y_gamma _p[9]
#define Y_tau0 _p[10]
#define gion _p[11]
#define Xinf _p[12]
#define Xtau _p[13]
#define Yinf _p[14]
#define Ytau _p[15]
#define Zinf _p[16]
#define Ztau _p[17]
#define X _p[18]
#define Y _p[19]
#define Z _p[20]
#define ena _p[21]
#define ina _p[22]
#define DX _p[23]
#define DY _p[24]
#define DZ _p[25]
#define v _p[26]
#define _g _p[27]
#define _ion_ena	*_ppvar[0]._pval
#define _ion_ina	*_ppvar[1]._pval
#define _ion_dinadv	*_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_rates(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_Na_BG_axon", _hoc_setdata,
 "rates_Na_BG_axon", _hoc_rates,
 0, 0
};
 
static void _check_rates(double*, Datum*, Datum*, _NrnThread*); 
static void _check_table_thread(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, int _type) {
   _check_rates(_p, _ppvar, _thread, _nt);
 }
 /* declare global and static user variables */
#define usetable usetable_Na_BG_axon
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_Na_BG_axon", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gmax_Na_BG_axon", "S/cm2",
 "gion_Na_BG_axon", "S/cm2",
 "Xtau_Na_BG_axon", "ms",
 "Ytau_Na_BG_axon", "ms",
 "Ztau_Na_BG_axon", "ms",
 0,0
};
 static double X0 = 0;
 static double Y0 = 0;
 static double Z0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "usetable_Na_BG_axon", &usetable_Na_BG_axon,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(_NrnThread*, _Memb_list*, int);
static void _ode_matsol(_NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"Na_BG_axon",
 "gmax_Na_BG_axon",
 "X_v0_Na_BG_axon",
 "X_k0_Na_BG_axon",
 "X_kt_Na_BG_axon",
 "X_gamma_Na_BG_axon",
 "X_tau0_Na_BG_axon",
 "Y_v0_Na_BG_axon",
 "Y_k0_Na_BG_axon",
 "Y_kt_Na_BG_axon",
 "Y_gamma_Na_BG_axon",
 "Y_tau0_Na_BG_axon",
 0,
 "gion_Na_BG_axon",
 "Xinf_Na_BG_axon",
 "Xtau_Na_BG_axon",
 "Yinf_Na_BG_axon",
 "Ytau_Na_BG_axon",
 "Zinf_Na_BG_axon",
 "Ztau_Na_BG_axon",
 0,
 "X_Na_BG_axon",
 "Y_Na_BG_axon",
 "Z_Na_BG_axon",
 0,
 0};
 static Symbol* _na_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 28, _prop);
 	/*initialize range parameters*/
 	gmax = 0.005;
 	X_v0 = -35;
 	X_k0 = 7;
 	X_kt = 20;
 	X_gamma = 0.45;
 	X_tau0 = 0.02;
 	Y_v0 = -50;
 	Y_k0 = -2;
 	Y_kt = 0.1;
 	Y_gamma = 0.2;
 	Y_tau0 = 0.5;
 	_prop->param = _p;
 	_prop->param_size = 28;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ena */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ina */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dinadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _Na_BG_axon_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("na", 1.0);
 	_na_sym = hoc_lookup("na_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
     _nrn_thread_table_reg(_mechtype, _check_table_thread);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 28, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Na_BG_axon /home/tarluca/optimizer_20220509/ca3_minimal_for_pygmo/mods/Na_BG_axon.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double *_t_Xinf;
 static double *_t_Xtau;
 static double *_t_Yinf;
 static double *_t_Ytau;
 static double *_t_Zinf;
 static double *_t_Ztau;
static int _reset;
static char *modelname = "Channel: Na_BG_axon";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_rates(_threadargsprotocomma_ double);
static int rates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static void _n_rates(_threadargsprotocomma_ double _lv);
 static int _slist1[3], _dlist1[3];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   DX = ( Xinf - X ) / Xtau ;
   DY = ( Yinf - Y ) / Ytau ;
   DZ = ( Zinf - Z ) / Ztau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 DX = DX  / (1. - dt*( ( ( ( - 1.0 ) ) ) / Xtau )) ;
 DY = DY  / (1. - dt*( ( ( ( - 1.0 ) ) ) / Ytau )) ;
 DZ = DZ  / (1. - dt*( ( ( ( - 1.0 ) ) ) / Ztau )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    X = X + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / Xtau)))*(- ( ( ( Xinf ) ) / Xtau ) / ( ( ( ( - 1.0 ) ) ) / Xtau ) - X) ;
    Y = Y + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / Ytau)))*(- ( ( ( Yinf ) ) / Ytau ) / ( ( ( ( - 1.0 ) ) ) / Ytau ) - Y) ;
    Z = Z + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / Ztau)))*(- ( ( ( Zinf ) ) / Ztau ) / ( ( ( ( - 1.0 ) ) ) / Ztau ) - Z) ;
   }
  return 0;
}
 static double _mfac_rates, _tmin_rates;
  static void _check_rates(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_celsius;
  static double _sav_X_v0;
  static double _sav_X_k0;
  static double _sav_X_kt;
  static double _sav_X_gamma;
  static double _sav_X_tau0;
  static double _sav_Y_v0;
  static double _sav_Y_k0;
  static double _sav_Y_kt;
  static double _sav_Y_gamma;
  static double _sav_Y_tau0;
  if (!usetable) {return;}
  if (_sav_celsius != celsius) { _maktable = 1;}
  if (_sav_X_v0 != X_v0) { _maktable = 1;}
  if (_sav_X_k0 != X_k0) { _maktable = 1;}
  if (_sav_X_kt != X_kt) { _maktable = 1;}
  if (_sav_X_gamma != X_gamma) { _maktable = 1;}
  if (_sav_X_tau0 != X_tau0) { _maktable = 1;}
  if (_sav_Y_v0 != Y_v0) { _maktable = 1;}
  if (_sav_Y_k0 != Y_k0) { _maktable = 1;}
  if (_sav_Y_kt != Y_kt) { _maktable = 1;}
  if (_sav_Y_gamma != Y_gamma) { _maktable = 1;}
  if (_sav_Y_tau0 != Y_tau0) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_rates =  - 100.0 ;
   _tmax =  50.0 ;
   _dx = (_tmax - _tmin_rates)/3000.; _mfac_rates = 1./_dx;
   for (_i=0, _x=_tmin_rates; _i < 3001; _x += _dx, _i++) {
    _f_rates(_p, _ppvar, _thread, _nt, _x);
    _t_Xinf[_i] = Xinf;
    _t_Xtau[_i] = Xtau;
    _t_Yinf[_i] = Yinf;
    _t_Ytau[_i] = Ytau;
    _t_Zinf[_i] = Zinf;
    _t_Ztau[_i] = Ztau;
   }
   _sav_celsius = celsius;
   _sav_X_v0 = X_v0;
   _sav_X_k0 = X_k0;
   _sav_X_kt = X_kt;
   _sav_X_gamma = X_gamma;
   _sav_X_tau0 = X_tau0;
   _sav_Y_v0 = Y_v0;
   _sav_Y_k0 = Y_k0;
   _sav_Y_kt = Y_kt;
   _sav_Y_gamma = Y_gamma;
   _sav_Y_tau0 = Y_tau0;
  }
 }

 static int rates(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _lv) { 
#if 0
_check_rates(_p, _ppvar, _thread, _nt);
#endif
 _n_rates(_p, _ppvar, _thread, _nt, _lv);
 return 0;
 }

 static void _n_rates(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_rates(_p, _ppvar, _thread, _nt, _lv); return; 
}
 _xi = _mfac_rates * (_lv - _tmin_rates);
 if (isnan(_xi)) {
  Xinf = _xi;
  Xtau = _xi;
  Yinf = _xi;
  Ytau = _xi;
  Zinf = _xi;
  Ztau = _xi;
  return;
 }
 if (_xi <= 0.) {
 Xinf = _t_Xinf[0];
 Xtau = _t_Xtau[0];
 Yinf = _t_Yinf[0];
 Ytau = _t_Ytau[0];
 Zinf = _t_Zinf[0];
 Ztau = _t_Ztau[0];
 return; }
 if (_xi >= 3000.) {
 Xinf = _t_Xinf[3000];
 Xtau = _t_Xtau[3000];
 Yinf = _t_Yinf[3000];
 Ytau = _t_Ytau[3000];
 Zinf = _t_Zinf[3000];
 Ztau = _t_Ztau[3000];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 Xinf = _t_Xinf[_i] + _theta*(_t_Xinf[_i+1] - _t_Xinf[_i]);
 Xtau = _t_Xtau[_i] + _theta*(_t_Xtau[_i+1] - _t_Xtau[_i]);
 Yinf = _t_Yinf[_i] + _theta*(_t_Yinf[_i+1] - _t_Yinf[_i]);
 Ytau = _t_Ytau[_i] + _theta*(_t_Ytau[_i+1] - _t_Ytau[_i]);
 Zinf = _t_Zinf[_i] + _theta*(_t_Zinf[_i+1] - _t_Zinf[_i]);
 Ztau = _t_Ztau[_i] + _theta*(_t_Ztau[_i+1] - _t_Ztau[_i]);
 }

 
static int  _f_rates ( _threadargsprotocomma_ double _lv ) {
   double _lalpha , _lbeta , _ltau , _linf , _ltemp_adj_X , _ltemp_adj_Y , _ltemp_adj_Z , _lA_alpha_Z , _lB_alpha_Z , _lVhalf_alpha_Z , _lA_beta_Z , _lB_beta_Z , _lVhalf_beta_Z ;
  _ltemp_adj_X = 1.0 ;
   _ltemp_adj_Y = 1.0 ;
   _ltemp_adj_Z = 1.0 ;
   _ltau = 1.0 / ( ( X_kt * ( exp ( X_gamma * ( _lv - X_v0 ) / X_k0 ) ) ) + ( X_kt * ( exp ( ( X_gamma - 1.0 ) * ( _lv - X_v0 ) / X_k0 ) ) ) ) + X_tau0 ;
   Xtau = _ltau / _ltemp_adj_X ;
   _linf = 1.0 / ( 1.0 + exp ( - ( _lv - X_v0 ) / X_k0 ) ) ;
   Xinf = _linf ;
   _ltau = 1.0 / ( ( Y_kt * ( exp ( Y_gamma * ( _lv - Y_v0 ) / Y_k0 ) ) ) + ( Y_kt * ( exp ( ( Y_gamma - 1.0 ) * ( _lv - Y_v0 ) / Y_k0 ) ) ) ) + Y_tau0 ;
   Ytau = _ltau / _ltemp_adj_Y ;
   _linf = 1.0 / ( 1.0 + exp ( - ( _lv - Y_v0 ) / Y_k0 ) ) ;
   Yinf = _linf ;
   _lv = _lv * 0.0010 ;
   _lalpha = ( 1.0 + 0.7 * ( exp ( ( _lv + 0.03 ) / 0.002 ) ) ) / ( 1.0 + ( exp ( ( _lv + 0.03 ) / 0.002 ) ) ) * ( 1.0 + ( exp ( 450.0 * ( _lv + 0.045 ) ) ) ) / ( 5.0 * ( exp ( 90.0 * ( _lv + 0.045 ) ) ) + 0.002 * ( exp ( 450.0 * ( _lv + 0.045 ) ) ) ) ;
   _lalpha = _lalpha * 0.0010 ;
   _lbeta = ( 1.0 + ( exp ( 450.0 * ( _lv + 0.045 ) ) ) ) / ( 5.0 * ( exp ( 90.0 * ( _lv + 0.045 ) ) ) + 0.002 * ( exp ( 450.0 * ( _lv + 0.045 ) ) ) ) - ( 1.0 + 0.7 * ( exp ( ( _lv + 0.03 ) / 0.002 ) ) ) / ( 1.0 + ( exp ( ( _lv + 0.03 ) / 0.002 ) ) ) * ( 1.0 + ( exp ( 450.0 * ( _lv + 0.045 ) ) ) ) / ( 5.0 * ( exp ( 90.0 * ( _lv + 0.045 ) ) ) + 0.002 * ( exp ( 450.0 * ( _lv + 0.045 ) ) ) ) ;
   _lbeta = _lbeta * 0.0010 ;
   _lv = _lv * 1000.0 ;
   Ztau = 1.0 / ( _ltemp_adj_Z * ( _lalpha + _lbeta ) ) ;
   Zinf = _lalpha / ( _lalpha + _lbeta ) ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 
#if 1
 _check_rates(_p, _ppvar, _thread, _nt);
#endif
 _r = 1.;
 rates ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 3;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ena = _ion_ena;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 3; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ena = _ion_ena;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_na_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_na_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_na_sym, _ppvar, 2, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  int _i; double _save;{
  X = X0;
  Y = Y0;
  Z = Z0;
 {
   ena = 55.0 ;
   rates ( _threadargscomma_ v ) ;
   X = Xinf ;
   Y = Yinf ;
   Z = Zinf ;
   }
 
}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];

#if 0
 _check_rates(_p, _ppvar, _thread, _nt);
#endif
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  ena = _ion_ena;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gion = gmax * ( pow( ( X ) , 3.0 ) ) * ( pow( ( Y ) , 1.0 ) ) * ( pow( ( Z ) , 1.0 ) ) ;
   ina = gion * ( v - ena ) ;
   }
 _current += ina;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  ena = _ion_ena;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dina;
  _dina = ina;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dinadv += (_dina - ina)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ina += ina ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  ena = _ion_ena;
 {   states(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(X) - _p;  _dlist1[0] = &(DX) - _p;
 _slist1[1] = &(Y) - _p;  _dlist1[1] = &(DY) - _p;
 _slist1[2] = &(Z) - _p;  _dlist1[2] = &(DZ) - _p;
   _t_Xinf = makevector(3001*sizeof(double));
   _t_Xtau = makevector(3001*sizeof(double));
   _t_Yinf = makevector(3001*sizeof(double));
   _t_Ytau = makevector(3001*sizeof(double));
   _t_Zinf = makevector(3001*sizeof(double));
   _t_Ztau = makevector(3001*sizeof(double));
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/tarluca/optimizer_20220509/ca3_minimal_for_pygmo/mods/Na_BG_axon.mod";
static const char* nmodl_file_text = 
  "TITLE Channel: Na_BG_axon\n"
  "\n"
  "COMMENT\n"
  "    Generic HH-type Na channel model in Borg-Graham format\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "UNITS {\n"
  "    (mA) = (milliamp)\n"
  "    (mV) = (millivolt)\n"
  "    (S) = (siemens)\n"
  "    (um) = (micrometer)\n"
  "    (molar) = (1/liter)\n"
  "    (mM) = (millimolar)\n"
  "    (l) = (liter)\n"
  "}\n"
  "\n"
  "\n"
  "NEURON {\n"
  "    \n"
  "    SUFFIX Na_BG_axon\n"
  "    USEION na READ ena WRITE ina VALENCE 1  ? reversal potential of ion is read, outgoing current is written\n"
  "           \n"
  "        \n"
  "    RANGE gmax, gion\n"
  "    \n"
  "    RANGE Xinf, Xtau\n"
  "    RANGE Yinf, Ytau\n"
  "	RANGE Zinf, Ztau\n"
  "        \n"
  "    RANGE X_v0, X_k0, X_kt, X_gamma, X_tau0\n"
  "    RANGE Y_v0, Y_k0, Y_kt, Y_gamma, Y_tau0\n"
  "\n"
  "}\n"
  "\n"
  "PARAMETER { \n"
  "    \n"
  "    gmax = 0.0050 (S/cm2)  ? default value, should be overwritten when conductance placed on cell\n"
  "    \n"
  "    X_v0 = -35.0 : Note units of this will be determined by its usage in the generic functions (mV)\n"
  "\n"
  "    X_k0 = 7 : Note units of this will be determined by its usage in the generic functions (mV)\n"
  "\n"
  "    X_kt = 20 : Note units of this will be determined by its usage in the generic functions (1/ms)\n"
  "\n"
  "    X_gamma = 0.45 : Note units of this will be determined by its usage in the generic functions\n"
  "\n"
  "    X_tau0 = 0.02 : Note units of this will be determined by its usage in the generic functions (ms)\n"
  "    \n"
  "    Y_v0 = -50 : Note units of this will be determined by its usage in the generic functions (mV)\n"
  "\n"
  "    Y_k0 = -2 : Note units of this will be determined by its usage in the generic functions (mV)\n"
  "\n"
  "    Y_kt = 0.1 : Note units of this will be determined by its usage in the generic functions (1/ms)\n"
  "\n"
  "    Y_gamma = 0.2 : Note units of this will be determined by its usage in the generic functions\n"
  "\n"
  "    Y_tau0 = 0.5 : Note units of this will be determined by its usage in the generic functions (ms)\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "    \n"
  "    v (mV)\n"
  "    \n"
  "    celsius (degC)\n"
  "          \n"
  "\n"
  "    ? Reversal potential of na\n"
  "    ena (mV)\n"
  "    ? The outward flow of ion: na calculated by rate equations...\n"
  "    ina (mA/cm2)\n"
  "    \n"
  "    \n"
  "    gion (S/cm2)\n"
  "    Xinf\n"
  "    Xtau (ms)\n"
  "    Yinf\n"
  "    Ytau (ms)\n"
  "    Zinf\n"
  "    Ztau (ms)\n"
  "}\n"
  "\n"
  "BREAKPOINT { \n"
  "                        \n"
  "    SOLVE states METHOD cnexp\n"
  "         \n"
  "\n"
  "    gion = gmax*((X)^3)*((Y)^1)*((Z)^1)\n"
  "\n"
  "    ina = gion*(v - ena)\n"
  "            \n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "INITIAL {\n"
  "    \n"
  "    ena = 55\n"
  "        \n"
  "    rates(v)\n"
  "    X = Xinf\n"
  "    Y = Yinf\n"
  "    Z = Zinf\n"
  "    \n"
  "}\n"
  "    \n"
  "STATE {\n"
  "    X\n"
  "    Y\n"
  "	Z\n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "    rates(v)\n"
  "    X' = (Xinf - X)/Xtau\n"
  "    Y' = (Yinf - Y)/Ytau\n"
  "	Z' = (Zinf - Z)/Ztau\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v(mV)) {  \n"
  "    \n"
  "    LOCAL alpha, beta, tau, inf, temp_adj_X, temp_adj_Y, temp_adj_Z, A_alpha_Z, B_alpha_Z, Vhalf_alpha_Z, A_beta_Z, B_beta_Z, Vhalf_beta_Z\n"
  "        \n"
  "    TABLE Xinf, Xtau,Yinf, Ytau,Zinf, Ztau\n"
  "    DEPEND celsius, X_v0, X_k0, X_kt, X_gamma, X_tau0, Y_v0, Y_k0, Y_kt, Y_gamma, Y_tau0\n"
  "    FROM -100 TO 50 WITH 3000\n"
  "    \n"
  "    \n"
  "    UNITSOFF\n"
  "\n"
  "    temp_adj_X = 1\n"
  "    temp_adj_Y = 1\n"
  "	temp_adj_Z = 1\n"
  "\n"
  "        \n"
  "    ?      ***  Adding rate equations for gate: X  ***\n"
  "            \n"
  "    tau = 1 / ( (X_kt * (exp (X_gamma * (v - X_v0) / X_k0))) + (X_kt * (exp ((X_gamma - 1)  * (v - X_v0) / X_k0)))) + X_tau0\n"
  "        \n"
  "    Xtau = tau/temp_adj_X\n"
  "    \n"
  "    \n"
  "    inf = 1 / ( 1 + exp (-(v - X_v0) / X_k0)) \n"
  "        \n"
  "    Xinf = inf\n"
  "    \n"
  "    ?     *** Finished rate equations for gate: X ***\n"
  "    \n"
  "    \n"
  "    ?      ***  Adding rate equations for gate: Y  ***\n"
  "    \n"
  "    tau = 1 / ( (Y_kt * (exp (Y_gamma * (v - Y_v0) / Y_k0))) + (Y_kt * (exp ((Y_gamma - 1)  * (v - Y_v0) / Y_k0)))) + Y_tau0\n"
  "    \n"
  "    Ytau = tau/temp_adj_Y\n"
  "    \n"
  "    \n"
  "    inf = 1 / ( 1 + exp (-(v - Y_v0) / Y_k0)) \n"
  "    \n"
  "    Yinf = inf\n"
  "    \n"
  "    ?     *** Finished rate equations for gate: Y ***\n"
  "    \n"
  "	\n"
  "	\n"
  "	?      ***  Adding rate equations for gate: Z  ***\n"
  "    \n"
  "    ? Note: Equation (and all ChannelML file values) in SI Units so need to convert v first...\n"
  "    \n"
  "    v = v * 0.0010   ? temporarily set v to units of equation...\n"
  "            \n"
  "    alpha = (1+0.7*( exp ((v+0.03)/0.002)))/(1+( exp ((v+0.03)/0.002)))*(1+(exp (450*(v+0.045))))/(5*(exp (90*(v+0.045))) + 0.002*(exp (450*(v+0.045))))\n"
  "        \n"
  "    ? Set correct units of alpha for NEURON\n"
  "    alpha = alpha * 0.0010 \n"
  "    \n"
  "    beta = (1+(exp (450*(v+0.045))))/(5*(exp (90*(v+0.045))) + 0.002*(exp (450*(v+0.045)))) - (1+0.7*( exp ((v+0.03)/0.002)))/(1+( exp ((v+0.03)/0.002)))*(1+(exp (450*(v+0.045))))/(5*(exp (90*(v+0.045))) + 0.002*(exp (450*(v+0.045))))\n"
  "        \n"
  "    ? Set correct units of beta for NEURON\n"
  "    beta = beta * 0.0010 \n"
  "    \n"
  "    v = v * 1000   ? reset v\n"
  "        \n"
  "    Ztau = 1/(temp_adj_Z*(alpha + beta))\n"
  "    Zinf = alpha/(alpha + beta)\n"
  "    \n"
  "    ?     *** Finished rate equations for gate: Z ***\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "UNITSON\n"
  "\n"
  "\n"
  ;
#endif
