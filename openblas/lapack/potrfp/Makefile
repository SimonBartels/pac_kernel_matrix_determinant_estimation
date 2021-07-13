TOPDIR	= ../..
include ../../Makefile.system


SBLASOBJS = #spotrfp_U_single.$(SUFFIX) spotrfp_L_single.$(SUFFIX) spotf2p_U.$(SUFFIX) spotf2p_L.$(SUFFIX)
DBLASOBJS = dpotrfp_U_single.$(SUFFIX) dpotrfp_L_single.$(SUFFIX) dpotf2p_U.$(SUFFIX) dpotf2p_L.$(SUFFIX) dpotrfp_L.$(SUFFIX) dpotrfp_L_debug.$(SUFFIX)
QBLASOBJS = #qpotrfp_U_single.$(SUFFIX) qpotrfp_L_single.$(SUFFIX) qpotf2p_U.$(SUFFIX) qpotf2p_L.$(SUFFIX)
CBLASOBJS = #cpotrfp_U_single.$(SUFFIX) cpotrfp_L_single.$(SUFFIX) cpotf2p_U.$(SUFFIX) cpotf2p_L.$(SUFFIX)
ZBLASOBJS = #zpotrfp_U_single.$(SUFFIX) zpotrfp_L_single.$(SUFFIX) zpotf2p_U.$(SUFFIX) zpotf2p_L.$(SUFFIX)
XBLASOBJS = #xpotrfp_U_single.$(SUFFIX) xpotrfp_L_single.$(SUFFIX) xpotf2p_U.$(SUFFIX) xpotf2p_L.$(SUFFIX)

ifdef SMP
SBLASOBJS += #spotrfp_U_parallel.$(SUFFIX) spotrfp_L_parallel.$(SUFFIX)
DBLASOBJS += dpotrfp_U_parallel.$(SUFFIX) dpotrfp_L_parallel.$(SUFFIX)
QBLASOBJS += #qpotrfp_U_parallel.$(SUFFIX) qpotrfp_L_parallel.$(SUFFIX)
CBLASOBJS += #cpotrfp_U_parallel.$(SUFFIX) cpotrfp_L_parallel.$(SUFFIX)
ZBLASOBJS += #zpotrfp_U_parallel.$(SUFFIX) zpotrfp_L_parallel.$(SUFFIX)
XBLASOBJS += #xpotrfp_U_parallel.$(SUFFIX) xpotrfp_L_parallel.$(SUFFIX)
endif

dpotrfp_L.$(SUFFIX) : potrfp_L.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -DDOUBLE $< -o $(@F)

dpotrfp_L.$(PSUFFIX) : potrfp_L.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -DDOUBLE $< -o $(@F)
	
dpotrfp_L_debug.$(SUFFIX) : potrfp_L_debug.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -DDOUBLE $< -o $(@F)

dpotrfp_L_debug.$(PSUFFIX) : potrfp_L_debug.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -DDOUBLE $< -o $(@F)



spotf2p_U.$(SUFFIX) : potf2p_U.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -UDOUBLE $< -o $(@F)

spotf2p_L.$(SUFFIX) : potf2p_L.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -UDOUBLE $< -o $(@F)

dpotf2p_U.$(SUFFIX) : potf2p_U.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -DDOUBLE $< -o $(@F)

dpotf2p_L.$(SUFFIX) : potf2p_L.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -DDOUBLE $< -o $(@F)

qpotf2p_U.$(SUFFIX) : potf2p_U.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -DXDOUBLE $< -o $(@F)

qpotf2p_L.$(SUFFIX) : potf2p_L.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -DXDOUBLE $< -o $(@F)

# TODO: find solution for complex matrices
cpotf2p_U.$(SUFFIX) : potf2p_U.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -UDOUBLE $< -o $(@F)

cpotf2p_L.$(SUFFIX) : potf2p_L.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -UDOUBLE $< -o $(@F)

zpotf2p_U.$(SUFFIX) : potf2p_U.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -DDOUBLE $< -o $(@F)

zpotf2p_L.$(SUFFIX) : potf2p_L.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -DDOUBLE $< -o $(@F)

xpotf2p_U.$(SUFFIX) : potf2p_U.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -DXDOUBLE $< -o $(@F)

xpotf2p_L.$(SUFFIX) : potf2p_L.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -DXDOUBLE $< -o $(@F)

spotf2p_U.$(PSUFFIX) : potf2p_U.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -UDOUBLE $< -o $(@F)

spotf2p_L.$(PSUFFIX) : potf2p_L.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -UDOUBLE $< -o $(@F)

dpotf2p_U.$(PSUFFIX) : potf2p_U.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -DDOUBLE $< -o $(@F)

dpotf2p_L.$(PSUFFIX) : potf2p_L.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -DDOUBLE $< -o $(@F)

qpotf2p_U.$(PSUFFIX) : potf2p_U.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -DXDOUBLE $< -o $(@F)

qpotf2p_L.$(PSUFFIX) : potf2p_L.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -DXDOUBLE $< -o $(@F)

# TODO: Find a solution for the complex case.
cpotf2p_U.$(PSUFFIX) : potf2p_U.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -UDOUBLE $< -o $(@F)

cpotf2p_L.$(PSUFFIX) : potf2p_L.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -UDOUBLE $< -o $(@F)

zpotf2p_U.$(PSUFFIX) : potf2p_U.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -DDOUBLE $< -o $(@F)

zpotf2p_L.$(PSUFFIX) : potf2p_L.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -DDOUBLE $< -o $(@F)

xpotf2p_U.$(PSUFFIX) : potf2p_U.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -DXDOUBLE $< -o $(@F)

xpotf2p_L.$(PSUFFIX) : potf2p_L.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -DXDOUBLE $< -o $(@F)

spotrfp_U_single.$(SUFFIX) : potrfp_U_single.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -UDOUBLE $< -o $(@F)

spotrfp_L_single.$(SUFFIX) : potrfp_L_single.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -UDOUBLE $< -o $(@F)

spotrfp_U_parallel.$(SUFFIX) : potrfp_U_parallel.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -UDOUBLE $< -o $(@F)

spotrfp_L_parallel.$(SUFFIX) : potrfp_L_parallel.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -UDOUBLE $< -o $(@F)

dpotrfp_U_single.$(SUFFIX) : potrfp_U_single.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -DDOUBLE $< -o $(@F)

dpotrfp_L_single.$(SUFFIX) : potrfp_L_single.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -DDOUBLE $< -o $(@F)

dpotrfp_U_parallel.$(SUFFIX) : potrfp_U_parallel.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -DDOUBLE $< -o $(@F)

dpotrfp_L_parallel.$(SUFFIX) : potrfp_L_parallel.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -DDOUBLE $< -o $(@F)

qpotrfp_U_single.$(SUFFIX) : potrfp_U_single.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -DXDOUBLE $< -o $(@F)

qpotrfp_L_single.$(SUFFIX) : potrfp_L_single.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -DXDOUBLE $< -o $(@F)

qpotrfp_U_parallel.$(SUFFIX) : potrfp_U_parallel.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -DXDOUBLE $< -o $(@F)

qpotrfp_L_parallel.$(SUFFIX) : potrfp_L_parallel.c
	$(CC) -c $(CFLAGS) -UCOMPLEX -DXDOUBLE $< -o $(@F)

cpotrfp_U_single.$(SUFFIX) : potrfp_U_single.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -UDOUBLE $< -o $(@F)

cpotrfp_L_single.$(SUFFIX) : potrfp_L_single.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -UDOUBLE $< -o $(@F)

cpotrfp_U_parallel.$(SUFFIX) : potrfp_U_parallel.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -UDOUBLE $< -o $(@F)

cpotrfp_L_parallel.$(SUFFIX) : potrfp_L_parallel.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -UDOUBLE $< -o $(@F)

zpotrfp_U_single.$(SUFFIX) : potrfp_U_single.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -DDOUBLE $< -o $(@F)

zpotrfp_L_single.$(SUFFIX) : potrfp_L_single.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -DDOUBLE $< -o $(@F)

zpotrfp_U_parallel.$(SUFFIX) : potrfp_U_parallel.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -DDOUBLE $< -o $(@F)

zpotrfp_L_parallel.$(SUFFIX) : potrfp_L_parallel.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -DDOUBLE $< -o $(@F)

xpotrfp_U_single.$(SUFFIX) : potrfp_U_single.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -DXDOUBLE $< -o $(@F)

xpotrfp_L_single.$(SUFFIX) : potrfp_L_single.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -DXDOUBLE $< -o $(@F)

xpotrfp_U_parallel.$(SUFFIX) : potrfp_U_parallel.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -DXDOUBLE $< -o $(@F)

xpotrfp_L_parallel.$(SUFFIX) : potrfp_L_parallel.c
	$(CC) -c $(CFLAGS) -DCOMPLEX -DXDOUBLE $< -o $(@F)

spotrfp_U_single.$(PSUFFIX) : potrfp_U_single.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -UDOUBLE $< -o $(@F)

spotrfp_L_single.$(PSUFFIX) : potrfp_L_single.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -UDOUBLE $< -o $(@F)

spotrfp_U_parallel.$(PSUFFIX) : potrfp_U_parallel.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -UDOUBLE $< -o $(@F)

spotrfp_L_parallel.$(PSUFFIX) : potrfp_L_parallel.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -UDOUBLE $< -o $(@F)

dpotrfp_U_single.$(PSUFFIX) : potrfp_U_single.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -DDOUBLE $< -o $(@F)

dpotrfp_L_single.$(PSUFFIX) : potrfp_L_single.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -DDOUBLE $< -o $(@F)

dpotrfp_U_parallel.$(PSUFFIX) : potrfp_U_parallel.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -DDOUBLE $< -o $(@F)

dpotrfp_L_parallel.$(PSUFFIX) : potrfp_L_parallel.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -DDOUBLE $< -o $(@F)

qpotrfp_U_single.$(PSUFFIX) : potrfp_U_single.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -DXDOUBLE $< -o $(@F)

qpotrfp_L_single.$(PSUFFIX) : potrfp_L_single.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -DXDOUBLE $< -o $(@F)

qpotrfp_U_parallel.$(PSUFFIX) : potrfp_U_parallel.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -DXDOUBLE $< -o $(@F)

qpotrfp_L_parallel.$(PSUFFIX) : potrfp_L_parallel.c
	$(CC) -c $(PFLAGS) -UCOMPLEX -DXDOUBLE $< -o $(@F)

cpotrfp_U_single.$(PSUFFIX) : potrfp_U_single.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -UDOUBLE $< -o $(@F)

cpotrfp_L_single.$(PSUFFIX) : potrfp_L_single.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -UDOUBLE $< -o $(@F)

cpotrfp_U_parallel.$(PSUFFIX) : potrfp_U_parallel.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -UDOUBLE $< -o $(@F)

cpotrfp_L_parallel.$(PSUFFIX) : potrfp_L_parallel.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -UDOUBLE $< -o $(@F)

zpotrfp_U_single.$(PSUFFIX) : potrfp_U_single.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -DDOUBLE $< -o $(@F)

zpotrfp_L_single.$(PSUFFIX) : potrfp_L_single.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -DDOUBLE $< -o $(@F)

zpotrfp_U_parallel.$(PSUFFIX) : potrfp_U_parallel.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -DDOUBLE $< -o $(@F)

zpotrfp_L_parallel.$(PSUFFIX) : potrfp_L_parallel.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -DDOUBLE $< -o $(@F)

xpotrfp_U_single.$(PSUFFIX) : potrfp_U_single.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -DXDOUBLE $< -o $(@F)

xpotrfp_L_single.$(PSUFFIX) : potrfp_L_single.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -DXDOUBLE $< -o $(@F)

xpotrfp_U_parallel.$(PSUFFIX) : potrfp_U_parallel.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -DXDOUBLE $< -o $(@F)

xpotrfp_L_parallel.$(PSUFFIX) : potrfp_L_parallel.c
	$(CC) -c $(PFLAGS) -DCOMPLEX -DXDOUBLE $< -o $(@F)

include ../../Makefile.tail
