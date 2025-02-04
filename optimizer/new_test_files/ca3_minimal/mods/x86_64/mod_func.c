#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _bk_ch_reg(void);
extern void _bk_ch_pool_reg(void);
extern void _cacum_lpool_reg(void);
extern void _cacum_reg(void);
extern void _cacum_npool_reg(void);
extern void _CaL_pool2_inact_reg(void);
extern void _CaL_pool2_inact_params_minimal_eca_reg(void);
extern void _CaL_pool2_inact_params_minimal_reg(void);
extern void _CaL_pool2_reg(void);
extern void _CaN_BG_pool2_reg(void);
extern void _H_CA1pyr_dist_reg(void);
extern void _H_CA1pyr_prox_reg(void);
extern void _K_A_dist_reg(void);
extern void _K_AHP3_reg(void);
extern void _K_A_prox_reg(void);
extern void _kd_params3_reg(void);
extern void _K_DRS4_params_voltage_dep_reg(void);
extern void _km_q10_2_reg(void);
extern void _Leak_pyr_reg(void);
extern void _Na_BG_axon_reg(void);
extern void _Na_BG_dend_reg(void);
extern void _Na_BG_soma_reg(void);
extern void _SK2_DP_reg(void);
extern void _sKCa_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," \"bk_ch.mod\"");
    fprintf(stderr," \"bk_ch_pool.mod\"");
    fprintf(stderr," \"cacum_lpool.mod\"");
    fprintf(stderr," \"cacum.mod\"");
    fprintf(stderr," \"cacum_npool.mod\"");
    fprintf(stderr," \"CaL_pool2_inact.mod\"");
    fprintf(stderr," \"CaL_pool2_inact_params_minimal_eca.mod\"");
    fprintf(stderr," \"CaL_pool2_inact_params_minimal.mod\"");
    fprintf(stderr," \"CaL_pool2.mod\"");
    fprintf(stderr," \"CaN_BG_pool2.mod\"");
    fprintf(stderr," \"H_CA1pyr_dist.mod\"");
    fprintf(stderr," \"H_CA1pyr_prox.mod\"");
    fprintf(stderr," \"K_A_dist.mod\"");
    fprintf(stderr," \"K_AHP3.mod\"");
    fprintf(stderr," \"K_A_prox.mod\"");
    fprintf(stderr," \"kd_params3.mod\"");
    fprintf(stderr," \"K_DRS4_params_voltage_dep.mod\"");
    fprintf(stderr," \"km_q10_2.mod\"");
    fprintf(stderr," \"Leak_pyr.mod\"");
    fprintf(stderr," \"Na_BG_axon.mod\"");
    fprintf(stderr," \"Na_BG_dend.mod\"");
    fprintf(stderr," \"Na_BG_soma.mod\"");
    fprintf(stderr," \"SK2_DP.mod\"");
    fprintf(stderr," \"sKCa.mod\"");
    fprintf(stderr, "\n");
  }
  _bk_ch_reg();
  _bk_ch_pool_reg();
  _cacum_lpool_reg();
  _cacum_reg();
  _cacum_npool_reg();
  _CaL_pool2_inact_reg();
  _CaL_pool2_inact_params_minimal_eca_reg();
  _CaL_pool2_inact_params_minimal_reg();
  _CaL_pool2_reg();
  _CaN_BG_pool2_reg();
  _H_CA1pyr_dist_reg();
  _H_CA1pyr_prox_reg();
  _K_A_dist_reg();
  _K_AHP3_reg();
  _K_A_prox_reg();
  _kd_params3_reg();
  _K_DRS4_params_voltage_dep_reg();
  _km_q10_2_reg();
  _Leak_pyr_reg();
  _Na_BG_axon_reg();
  _Na_BG_dend_reg();
  _Na_BG_soma_reg();
  _SK2_DP_reg();
  _sKCa_reg();
}
