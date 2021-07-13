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

//#include <stdio.h>
#include <math.h>
#include "common.h"


inline int potrfp_sign(FLOAT x) {
    return (x < 0) ? -1 : (x > 0);
}

inline FLOAT extract_log_det(FLOAT *aoffset, BLASLONG lda, BLASLONG n) {
    FLOAT ldet = 0;
    BLASLONG i;
    for (i = 0; i < n; i++)
        ldet += 2 * log(*(aoffset + (i * lda + i) * COMPSIZE));
    return ldet;
}

blasint CNAME(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n, FLOAT *sa, FLOAT *sb, BLASLONG myid,
              potrfp_constants *args2, potrfp_values *args3) {
    if (args2->method == 0) return default_chol(args, range_m, range_n, sa, sb, myid, args2, args3);
    if (args2->method == 1) return banachiewicz_chol(args, range_m, range_n, sa, sb, myid, args2, args3);
    return -1;
}

blasint banachiewicz_chol(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n, FLOAT *sa, FLOAT *sb, BLASLONG myid,
                          potrfp_constants *args2, potrfp_values *args3){
    BLASLONG n, bk, i, blocking, lda;
    BLASLONG info;
    int mode;
    blas_arg_t newarg;
    FLOAT *sub_det = &(args3->sub_det);

    FLOAT *a;
    FLOAT alpha[2] = {-ONE, ZERO};

#ifndef COMPLEX
#ifdef XDOUBLE
    mode  =  BLAS_XDOUBLE | BLAS_REAL;
#elif defined(DOUBLE)
    mode  =  BLAS_DOUBLE  | BLAS_REAL;
#else
    mode = BLAS_SINGLE | BLAS_REAL;
#endif
#else
#ifdef XDOUBLE
    mode  =  BLAS_XDOUBLE | BLAS_COMPLEX;
#elif defined(DOUBLE)
    mode  =  BLAS_DOUBLE  | BLAS_COMPLEX;
#else
    mode  =  BLAS_SINGLE  | BLAS_COMPLEX;
#endif
#endif

    n = args->n;
    a = (FLOAT *) args->a;
    lda = args->lda;

    *sub_det = 0.;
    args3->estimate = *sub_det;
    args3->hierarchy_level = 0; // return after how many datapoints we stopped
    args3->mean = 0.;

    if (range_n) n = range_n[1] - range_n[0];


    newarg.lda = lda;
    newarg.ldb = lda;
    newarg.ldc = lda;
    newarg.alpha = alpha;
    newarg.beta = NULL;
    newarg.nthreads = args->nthreads;

    //blocking = ((n / 2 + GEMM_UNROLL_N - 1)/GEMM_UNROLL_N) * GEMM_UNROLL_N;
    //if (blocking > GEMM_Q) blocking = GEMM_Q;
    blocking = args2->blocking;

    if (n < args2->initial_block) args2->initial_block = n;

    bk = n;
    if (bk > args2->initial_block) bk = args2->initial_block;
    newarg.m = bk;
    newarg.n = bk;
    newarg.a = a;
    info = POTRF_L_PARALLEL(&newarg, NULL, NULL, sa, sb, 0);

    if (info) return info;
    *sub_det += extract_log_det(a, lda, bk);
    for (i = bk; i < n; i += blocking) {
        bk = n - i;
        if (bk > blocking) bk = blocking;
        // check bound conditions
        FLOAT lbound = *sub_det + args2->lnSmallestEval * (n - i);
        FLOAT ubound = *sub_det + fmin(args2->C_Hinv + (n - i) * (*sub_det + args2->C_Hinv) / i,
                                       (n - i) * (args2->C + args2->lnSmallestEval));
        if ((potrfp_sign(lbound) * potrfp_sign(ubound)) == 1 &&
            (ubound - lbound <= 2 * args2->r * fmin(fabs(lbound), fabs(ubound)))) {
            args3->estimate = lbound / 2 + ubound / 2;
            args3->hierarchy_level = i; // return after how many datapoints we stopped
            args3->mean = *sub_det / i;
            return 0;
        }

        newarg.m = bk;
        newarg.n = i;
        newarg.a = a;
        newarg.b = a + i * COMPSIZE; // seems to do what I want
        gemm_thread_m(mode | BLAS_RSIDE | BLAS_TRANSA_T | BLAS_UPLO,
                      &newarg, NULL, NULL, (void *) TRSM_RCLN, sa, sb, args->nthreads);

        newarg.n = bk;
        newarg.m = bk;
        newarg.k = i; // k can not describe any properties of the target!
        newarg.a = newarg.b; // this must be the same as the previous newarg.b
        newarg.c = a + (i + i * lda) * COMPSIZE; // I want to write to the same target starting point

#ifndef USE_SIMPLE_THREADED_LEVEL3
        HERK_THREAD_LN(&newarg, NULL, NULL, sa, sb, 0);
#else
        syrk_thread(mode | BLAS_TRANSA_N | BLAS_TRANSB_T | BLAS_UPLO,
            &newarg, NULL, NULL, (void *)HERK_LN, sa, sb, args -> nthreads);
#endif
        newarg.m = bk;
        newarg.n = bk;
        newarg.a = newarg.c; // I want to compute the Cholesky just where I did the update

        info = POTRF_L_PARALLEL(&newarg, NULL, NULL, sa, sb, 0);
        if (info) return info + i;
        *sub_det += extract_log_det(newarg.a, lda, bk);
    }
    args3->estimate = *sub_det;
    args3->hierarchy_level = n; // return after how many datapoints we stopped
    args3->mean = *sub_det / n;
    return 0;
}


blasint default_chol(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n, FLOAT *sa, FLOAT *sb, BLASLONG myid,
                     potrfp_constants *args2, potrfp_values *args3) {
    BLASLONG n, bk, i, blocking, lda;
    BLASLONG info;
    int mode;
    blas_arg_t newarg;

    FLOAT *a;
    FLOAT alpha[2] = {-ONE, ZERO};

#ifndef COMPLEX
#ifdef XDOUBLE
    mode  =  BLAS_XDOUBLE | BLAS_REAL;
#elif defined(DOUBLE)
    mode  =  BLAS_DOUBLE  | BLAS_REAL;
#else
    mode = BLAS_SINGLE | BLAS_REAL;
#endif
#else
#ifdef XDOUBLE
    mode  =  BLAS_XDOUBLE | BLAS_COMPLEX;
#elif defined(DOUBLE)
    mode  =  BLAS_DOUBLE  | BLAS_COMPLEX;
#else
    mode  =  BLAS_SINGLE  | BLAS_COMPLEX;
#endif
#endif

    if (args->nthreads == 1) {
        info = POTRFP_L_SINGLE(args, NULL, NULL, sa, sb, 0, args2, args3);
        return info;
    }

    n = args->n;
    a = (FLOAT *) args->a;
    lda = args->lda;

    if (range_n) n = range_n[1] - range_n[0];

    if (n <= GEMM_UNROLL_N * 4) {
        info = POTRFP_L_SINGLE(args, NULL, range_n, sa, sb, 0, args2, args3);
        return info;
    }

    newarg.lda = lda;
    newarg.ldb = lda;
    newarg.ldc = lda;
    newarg.alpha = alpha;
    newarg.beta = NULL;
    newarg.nthreads = args->nthreads;

    //blocking = ((n / 2 + GEMM_UNROLL_N - 1)/GEMM_UNROLL_N) * GEMM_UNROLL_N;
    //if (blocking > GEMM_Q) blocking = GEMM_Q;
    blocking = args2->blocking;

    FLOAT *sub_det = &(args3->sub_det);
    *sub_det = 0.;

    for (i = 0; i < n; i += blocking) {
        bk = n - i;
        if (bk > blocking) bk = blocking;

        newarg.m = bk;
        newarg.n = bk;
        newarg.a = a + (i + i * lda) * COMPSIZE;

        info = POTRF_L_PARALLEL(&newarg, NULL, NULL, sa, sb, 0);
        if (info) return info + i;

        *sub_det += extract_log_det(newarg.a, lda, bk);

        if (n - i - bk > 0) {
            newarg.m = n - i - bk;
            newarg.n = bk;
            newarg.a = a + (i + i * lda) * COMPSIZE;
            newarg.b = a + (i + bk + i * lda) * COMPSIZE;

            gemm_thread_m(mode | BLAS_RSIDE | BLAS_TRANSA_T | BLAS_UPLO,
                          &newarg, NULL, NULL, (void *) TRSM_RCLN, sa, sb, args->nthreads);

            newarg.n = n - i - bk;
            newarg.k = bk;
            newarg.a = a + (i + bk + i * lda) * COMPSIZE;
            newarg.c = a + (i + bk + (i + bk) * lda) * COMPSIZE;

#ifndef USE_SIMPLE_THREADED_LEVEL3
            HERK_THREAD_LN(&newarg, NULL, NULL, sa, sb, 0);
#else
            syrk_thread(mode | BLAS_TRANSA_N | BLAS_TRANSB_T | BLAS_UPLO,
                &newarg, NULL, NULL, (void *)HERK_LN, sa, sb, args -> nthreads);
#endif

            // check bound conditions
            blasint step = i + bk;
            FLOAT lbound = *sub_det + args2->lnSmallestEval * (n - step);
            FLOAT ubound = *sub_det + fmin(args2->C_Hinv + (n - step) * (*sub_det + args2->C_Hinv) / step,
                                           (n - step) * (args2->C + args2->lnSmallestEval));
            if ((potrfp_sign(lbound) * potrfp_sign(ubound)) == 1 &&
                (ubound - lbound <= 2 * args2->r * fmin(fabs(lbound), fabs(ubound)))) {
                args3->estimate = lbound / 2 + ubound / 2;
                args3->hierarchy_level = step; // return after how many datapoints we stopped
                args3->mean = *sub_det / step;
                return 0;
            }
        }
    }
    args3->estimate = *sub_det;
    args3->hierarchy_level = n; // return after how many datapoints we stopped
    args3->mean = *sub_det / n;
    return 0;
}
