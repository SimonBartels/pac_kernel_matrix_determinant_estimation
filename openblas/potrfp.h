#ifndef POTRFP
#define POTRFP

typedef struct{
 // parameters
 FLOAT r;
 FLOAT C_Hinv; 
 BLASLONG blocking;
 BLASLONG initial_block;
 blasint method;
 
 // constants
 FLOAT lnSmallestEval; // ln(sn2)
 FLOAT C; // upper bound to the random variables
} potrfp_constants;

typedef struct{
 // how deep down we are in the recursiveness of the call
 // is used also as indicator how many datapoints we actually processed
 blasint hierarchy_level;

 // calculated values
 FLOAT mean;
 
 //return values
 FLOAT estimate;
 FLOAT sub_det; // the determinant of the current submatrix 
} potrfp_values;

#endif
