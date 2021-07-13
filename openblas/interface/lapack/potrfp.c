/*********************************************************************/
/* Copyright 2009, 2010 The University of Texas at Austin.           */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF    */
/*    MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE    */
/*    DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT    */
/*    AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,    */
/*    INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES    */
/*    (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE    */
/*    GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR    */
/*    BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF    */
/*    LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT    */
/*    (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT    */
/*    OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE    */
/*    POSSIBILITY OF SUCH DAMAGE.                                    */
/*                                                                   */
/* The views and conclusions contained in the software and           */
/* documentation are those of the authors and should not be          */
/* interpreted as representing official policies, either expressed   */
/* or implied, of The University of Texas at Austin.                 */
/*********************************************************************/

#include <stdio.h>
#include "common.h"
#ifdef FUNCTION_PROFILE
#include "functable.h"
#endif

#ifdef XDOUBLE
#define ERROR_NAME "QPOTRFP"
#elif defined(DOUBLE)
#define ERROR_NAME "DPOTRFP"
#else
#define ERROR_NAME "SPOTRFP"
#endif

//#include "potrfp.h"

static blasint (*potrfp_single[])(blas_arg_t *, BLASLONG *, BLASLONG *, FLOAT *, FLOAT *, BLASLONG, potrfp_constants *, potrfp_values *) = {
        POTRFP_L, POTRFP_L, POTRFP_L_DEBUG,
};

#ifdef SMP
static blasint (*potrfp_parallel[])(blas_arg_t *, BLASLONG *, BLASLONG *, FLOAT *, FLOAT *, BLASLONG, potrfp_constants *, potrfp_values *) = {
        POTRFP_L, POTRFP_L, POTRFP_L_DEBUG,
};
#endif

int NAME(char *UPLO, blasint *N, FLOAT *a, blasint *ldA, blasint *Info, FLOAT r, FLOAT C_Hinv, FLOAT lnSmallestEval, FLOAT C, BLASLONG blocking, BLASLONG initial_block, FLOAT *est, blasint *n_sub, FLOAT *mean){
  blas_arg_t args;
  potrfp_constants args2;
  potrfp_values args3;
  
  args2.r = r;
  args2.C_Hinv = C_Hinv;
  args2.lnSmallestEval = lnSmallestEval; //log(sn2)
  args2.C = C; //log(max(k(x,x)) + sn2) - args2.lnSmallestEval;
  args2.blocking = blocking;
  args2.initial_block = initial_block;
  args3.hierarchy_level = 0;
  args3.mean = 0.;
  args3.estimate = 0.;
  args3.sub_det = 0.;
  
  blasint uplo_arg = *UPLO;
  blasint uplo;
  blasint info;
  FLOAT *buffer;
#ifdef PPC440
  extern
#endif
  FLOAT *sa, *sb;

  PRINT_DEBUG_NAME;

  args.n    = *N;
  args.a    = (void *)a;
  args.lda  = *ldA;

  TOUPPER(uplo_arg);

  uplo = -1;
  if (uplo_arg == 'A') uplo = 0;  // BLAS_Default
  if (uplo_arg == 'B') uplo = 1;  // Banachiewicz
  if (uplo_arg == 'D') uplo = 2;  // Debug

  info  = 0;
  if (args.lda < MAX(1,args.n)) info = 4;
  if (args.n   < 0)             info = 2;
  if (uplo     < 0)             info = 1;
  if (info) {
    BLASFUNC(xerbla)(ERROR_NAME, &info, sizeof(ERROR_NAME) - 1);
    *Info = - info;
    return 0;
  }

  *Info = 0;

  if (args.n == 0) return 0;

  args2.method = uplo;

  IDEBUG_START;

  FUNCTION_PROFILE_START();

#ifndef PPC440
  buffer = (FLOAT *)blas_memory_alloc(1);

  sa = (FLOAT *)((BLASLONG)buffer + GEMM_OFFSET_A);
  sb = (FLOAT *)(((BLASLONG)sa + ((GEMM_P * GEMM_Q * COMPSIZE * SIZE + GEMM_ALIGN) & ~GEMM_ALIGN)) + GEMM_OFFSET_B);
#endif

#ifdef SMP
  args.common = NULL;
  args.nthreads = num_cpu_avail(4);

  if (args.nthreads == 1) {
#endif

    *Info = (potrfp_single[uplo])(&args, NULL, NULL, sa, sb, 0, &args2, &args3);

#ifdef SMP
  } else {
    *Info = (potrfp_parallel[uplo])(&args, NULL, NULL, sa, sb, 0, &args2, &args3);
  }
#endif

#ifndef PPC440
  blas_memory_free(buffer);
#endif

  FUNCTION_PROFILE_END(1, .5 * args.n * args.n,
		       args.n * (1./3. + args.n * ( 1./2. + args.n * 1./6.))
		       +  1./6. * args.n * (args.n * args.n - 1));

  IDEBUG_END;
  
  *est = args3.estimate;
  // this is taken care of in the main file
  //if (args3.hierarchy_level == 0 && **Info == 0) {
  //  args3.hierarchy_level = *N; // we did not stop early
  //  *est = args3.sub_det;
  //}
  *n_sub = args3.hierarchy_level;
  *mean = args3.mean;  
  return 0;
}
