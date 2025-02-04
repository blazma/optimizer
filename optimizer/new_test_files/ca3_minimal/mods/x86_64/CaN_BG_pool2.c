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
 
#define nrn_init _nrn_init__CaN_BG_pool2
#define _nrn_initial _nrn_initial__CaN_BG_pool2
#define nrn_cur _nrn_cur__CaN_BG_pool2
#define _nrn_current _nrn_current__CaN_BG_pool2
#define nrn_jacob _nrn_jacob__CaN_BG_pool2
#define nrn_state _nrn_state__CaN_BG_pool2
#define _net_receive _net_receive__CaN_BG_pool2 
#define _f_rates _f_rates__CaN_BG_pool2 
#define rates rates__CaN_BG_pool2 
#define states states__CaN_BG_pool2 
 
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
#define X _p[16]
#define Y _p[17]
#define eca _p[18]
#define ican _p[19]
#define ica _p[20]
#define DX _p[21]
#define DY _p[22]
#define v _p[23]
#define _g _p[24]
#define _ion_ica	*_ppvar[0]._pval
#define _ion_dicadv	*_ppvar[1]._pval
#define _ion_ican	*_ppvar[2]._pval
#define _ion_dicandv	*_ppvar[3]._pval
 
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
 "setdata_CaN_BG_pool2", _hoc_setdata,
 "rates_CaN_BG_pool2", _hoc_rates,
 0, 0
};
 
static void _check_rates(double*, Datum*, Datum*, _NrnThread*); 
static void _check_table_thread(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, int _type) {
   _check_rates(_p, _ppvar, _thread, _nt);
 }
 /* declare global and static user variables */
#define usetable usetable_CaN_BG_pool2
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_CaN_BG_pool2", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gmax_CaN_BG_pool2", "S/cm2",
 "gion_CaN_BG_pool2", "S/cm2",
 "Xtau_CaN_BG_pool2", "ms",
 "Ytau_CaN_BG_pool2", "ms",
 0,0
};
 static double X0 = 0;
 static double Y0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "usetable_CaN_BG_pool2", &usetable_CaN_BG_pool2,
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
 
#define _cvode_ieq _ppvar[4]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"CaN_BG_pool2",
 "gmax_CaN_BG_pool2",
 "X_v0_CaN_BG_pool2",
 "X_k0_CaN_BG_pool2",
 "X_kt_CaN_BG_pool2",
 "X_gamma_CaN_BG_pool2",
 "X_tau0_CaN_BG_pool2",
 "Y_v0_CaN_BG_pool2",
 "Y_k0_CaN_BG_pool2",
 "Y_kt_CaN_BG_pool2",
 "Y_gamma_CaN_BG_pool2",
 "Y_tau0_CaN_BG_pool2",
 0,
 "gion_CaN_BG_pool2",
 "Xinf_CaN_BG_pool2",
 "Xtau_CaN_BG_pool2",
 "Yinf_CaN_BG_pool2",
 "Ytau_CaN_BG_pool2",
 0,
 "X_CaN_BG_pool2",
 "Y_CaN_BG_pool2",
 0,
 0};
 static Symbol* _ca_sym;
 static Symbol* _can_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 25, _prop);
 	/*initialize range parameters*/
 	gmax = 0.002;
 	X_v0 = -4.5;
 	X_k0 = 6;
 	X_kt = 5;
 	X_gamma = 0.2;
 	X_tau0 = 0;
 	Y_v0 = -75;
 	Y_k0 = -6.5;
 	Y_kt = 1;
 	Y_gamma = 0.5;
 	Y_tau0 = 100;
 	_prop->param = _p;
 	_prop->param_size = 25;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 	_ppvar[0]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[1]._pval = &prop_ion->param[4]; /* _ion_dicadv */
 prop_ion = need_memb(_can_sym);
 	_ppvar[2]._pval = &prop_ion->param[3]; /* ican */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_dicandv */
 
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

 void _CaN_BG_pool2_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", 2.0);
 	ion_reg("can", 2.0);
 	_ca_sym = hoc_lookup("ca_ion");
 	_can_sym = hoc_lookup("can_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
     _nrn_thread_table_reg(_mechtype, _check_table_thread);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 25, 5);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "can_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "can_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 CaN_BG_pool2 /home/tarluca/optimizer_20220509/ca3_minimal_for_pygmo/mods/CaN_BG_pool2.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double *_t_Xinf;
 static double *_t_Xtau;
 static double *_t_Yinf;
 static double *_t_Ytau;
static int _reset;
static char *modelname = "Channel: CaN";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_rates(_threadargsprotocomma_ double);
static int rates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static void _n_rates(_threadargsprotocomma_ double _lv);
 static int _slist1[2], _dlist1[2];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   DX = ( Xinf - X ) / Xtau ;
   DY = ( Yinf - Y ) / Ytau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 DX = DX  / (1. - dt*( ( ( ( - 1.0 ) ) ) / Xtau )) ;
 DY = DY  / (1. - dt*( ( ( ( - 1.0 ) ) ) / Ytau )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    X = X + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / Xtau)))*(- ( ( ( Xinf ) ) / Xtau ) / ( ( ( ( - 1.0 ) ) ) / Xtau ) - X) ;
    Y = Y + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / Ytau)))*(- ( ( ( Yinf ) ) / Ytau ) / ( ( ( ( - 1.0 ) ) ) / Ytau ) - Y) ;
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
  return;
 }
 if (_xi <= 0.) {
 Xinf = _t_Xinf[0];
 Xtau = _t_Xtau[0];
 Yinf = _t_Yinf[0];
 Ytau = _t_Ytau[0];
 return; }
 if (_xi >= 3000.) {
 Xinf = _t_Xinf[3000];
 Xtau = _t_Xtau[3000];
 Yinf = _t_Yinf[3000];
 Ytau = _t_Ytau[3000];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 Xinf = _t_Xinf[_i] + _theta*(_t_Xinf[_i+1] - _t_Xinf[_i]);
 Xtau = _t_Xtau[_i] + _theta*(_t_Xtau[_i+1] - _t_Xtau[_i]);
 Yinf = _t_Yinf[_i] + _theta*(_t_Yinf[_i+1] - _t_Yinf[_i]);
 Ytau = _t_Ytau[_i] + _theta*(_t_Ytau[_i+1] - _t_Ytau[_i]);
 }

 
static int  _f_rates ( _threadargsprotocomma_ double _lv ) {
   double _ltau , _linf , _ltemp_adj_X , _ltemp_adj_Y ;
  _ltemp_adj_X = 1.0 ;
   _ltemp_adj_Y = 1.0 ;
   _ltau = 1.0 / ( ( X_kt * ( exp ( X_gamma * ( _lv - X_v0 ) / X_k0 ) ) ) + ( X_kt * ( exp ( ( X_gamma - 1.0 ) * ( _lv - X_v0 ) / X_k0 ) ) ) ) + X_tau0 ;
   Xtau = _ltau / _ltemp_adj_X ;
   _linf = 1.0 / ( 1.0 + exp ( - ( _lv - X_v0 ) / X_k0 ) ) ;
   Xinf = _linf ;
   _ltau = 1.0 / ( ( Y_kt * ( exp ( Y_gamma * ( _lv - Y_v0 ) / Y_k0 ) ) ) + ( Y_kt * ( exp ( ( Y_gamma - 1.0 ) * ( _lv - Y_v0 ) / Y_k0 ) ) ) ) + Y_tau0 ;
   Ytau = Y_tau0 / _ltemp_adj_Y ;
   _linf = 1.0 / ( 1.0 + exp ( - ( _lv - Y_v0 ) / Y_k0 ) ) ;
   Yinf = _linf ;
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
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 (_p, _ppvar, _thread, _nt);
   }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
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
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 1, 4);
   nrn_update_ion_pointer(_can_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_can_sym, _ppvar, 3, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  int _i; double _save;{
  X = X0;
  Y = Y0;
 {
   eca = 80.0 ;
   rates ( _threadargscomma_ v ) ;
   X = Xinf ;
   Y = Yinf ;
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
 initmodel(_p, _ppvar, _thread, _nt);
  }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gion = gmax * ( pow( ( X ) , 2.0 ) ) * ( pow( ( Y ) , 1.0 ) ) ;
   ican = gion * ( v - eca ) ;
   ica = ican ;
   }
 _current += ica;
 _current += ican;

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
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dican;
 double _dica;
  _dica = ica;
  _dican = ican;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dicadv += (_dica - ica)/.001 ;
  _ion_dicandv += (_dican - ican)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ica += ica ;
  _ion_ican += ican ;
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
 {   states(_p, _ppvar, _thread, _nt);
  }  }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(X) - _p;  _dlist1[0] = &(DX) - _p;
 _slist1[1] = &(Y) - _p;  _dlist1[1] = &(DY) - _p;
   _t_Xinf = makevector(3001*sizeof(double));
   _t_Xtau = makevector(3001*sizeof(double));
   _t_Yinf = makevector(3001*sizeof(double));
   _t_Ytau = makevector(3001*sizeof(double));
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/tarluca/optimizer_20220509/ca3_minimal_for_pygmo/mods/CaN_BG_pool2.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "\n"
  "   **************************************************\n"
  "   File generated by: neuroConstruct v1.5.1 \n"
  "   **************************************************\n"
  "\n"
  "\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "?  This is a NEURON mod file generated from a ChannelML file\n"
  "\n"
  "?  Unit system of original ChannelML file: SI Units\n"
  "\n"
  "COMMENT\n"
  "    ChannelML file containing a single Channel description\n"
  "ENDCOMMENT\n"
  "\n"
  "TITLE Channel: CaN\n"
  "\n"
  "COMMENT\n"
  "    High-threshold Ca(N) Channel in pyramid neurons\n"
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
  "    \n"
  "NEURON {\n"
  "      \n"
  "\n"
  "    SUFFIX CaN_BG_pool2\n"
  "	USEION ca WRITE ica VALENCE 2 ?  outgoing current is written\n"
  "    USEION can WRITE ican VALENCE 2 ?  outgoing current is written\n"
  "           \n"
  "        \n"
  "    RANGE gmax, gion\n"
  "    \n"
  "    RANGE Xinf, Xtau\n"
  "    \n"
  "    RANGE Yinf, Ytau\n"
  "	\n"
  "	RANGE X_v0, X_k0, X_kt, X_gamma, X_tau0\n"
  "    RANGE Y_v0, Y_k0, Y_kt, Y_gamma, Y_tau0\n"
  "    \n"
  "}\n"
  "\n"
  "PARAMETER { \n"
  "      \n"
  "\n"
  "    gmax = 0.0020 (S/cm2)  ? default value, should be overwritten when conductance placed on cell\n"
  "    \n"
  "	X_v0 = -4.5 : Note units of this will be determined by its usage in the generic functions (mV)\n"
  "\n"
  "    X_k0 = 6 : Note units of this will be determined by its usage in the generic functions (mV)\n"
  "\n"
  "    X_kt = 5 : Note units of this will be determined by its usage in the generic functions (1/ms)\n"
  "\n"
  "    X_gamma = 0.2 : Note units of this will be determined by its usage in the generic functions\n"
  "\n"
  "    X_tau0 = 0 : Note units of this will be determined by its usage in the generic functions (ms)\n"
  "    \n"
  "    Y_v0 = -75 : Note units of this will be determined by its usage in the generic functions (mV)\n"
  "\n"
  "    Y_k0 = -6.5 : Note units of this will be determined by its usage in the generic functions (mV)\n"
  "\n"
  "    Y_kt = 1: Note units of this will be determined by its usage in the generic functions (1/ms)\n"
  "\n"
  "    Y_gamma = 0.5 : Note units of this will be determined by its usage in the generic functions\n"
  "\n"
  "    Y_tau0 = 100 : Note units of this will be determined by its usage in the generic functions (ms)\n"
  "	\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "      \n"
  "\n"
  "    v (mV)\n"
  "    \n"
  "    celsius (degC)\n"
  "          \n"
  "\n"
  "    ? Reversal potential of ca\n"
  "    eca (mV)\n"
  "    ? The outward flow of ion: ca calculated by rate equations...\n"
  "    ican (mA/cm2)\n"
  "    ica (mA/cm2)\n"
  "    \n"
  "    gion (S/cm2)\n"
  "    Xinf\n"
  "    Xtau (ms)\n"
  "    Yinf\n"
  "    Ytau (ms)\n"
  "    \n"
  "}\n"
  "\n"
  "BREAKPOINT { \n"
  "                        \n"
  "    SOLVE states METHOD cnexp\n"
  "         \n"
  "\n"
  "    gion = gmax*((X)^2)*((Y)^1)      \n"
  "\n"
  "    ican = gion*(v - eca)\n"
  "	ica = ican\n"
  "            \n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "INITIAL {\n"
  "    \n"
  "    eca = 80\n"
  "        \n"
  "    rates(v)\n"
  "    X = Xinf\n"
  "        Y = Yinf\n"
  "        \n"
  "    \n"
  "}\n"
  "    \n"
  "STATE {\n"
  "    X\n"
  "    Y\n"
  "    \n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "    rates(v)\n"
  "    X' = (Xinf - X)/Xtau\n"
  "    Y' = (Yinf - Y)/Ytau\n"
  "    \n"
  "}\n"
  "\n"
  "PROCEDURE rates(v(mV)) {  \n"
  "    \n"
  "    ? Note: not all of these may be used, depending on the form of rate equations\n"
  "	LOCAL tau, inf, temp_adj_X, temp_adj_Y\n"
  "        \n"
  "    TABLE Xinf, Xtau,Yinf, Ytau\n"
  "    DEPEND celsius, X_v0, X_k0, X_kt, X_gamma, X_tau0, Y_v0, Y_k0, Y_kt, Y_gamma, Y_tau0\n"
  "    FROM -100 TO 50 WITH 3000\n"
  "    \n"
  "    \n"
  "    UNITSOFF\n"
  "\n"
  "    temp_adj_X = 1\n"
  "    temp_adj_Y = 1\n"
  "\n"
  "        \n"
  "   ?      ***  Adding rate equations for gate: X  ***\n"
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
  "    ? Ytau = tau/temp_adj_Y\n"
  "    \n"
  "	Ytau=Y_tau0/temp_adj_Y\n"
  "    \n"
  "    inf = 1 / ( 1 + exp (-(v - Y_v0) / Y_k0)) \n"
  "    \n"
  "    Yinf = inf\n"
  "    \n"
  "    ?     *** Finished rate equations for gate: Y ***\n"
  "\n"
  "         \n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "UNITSON\n"
  "\n"
  "\n"
  ;
#endif
