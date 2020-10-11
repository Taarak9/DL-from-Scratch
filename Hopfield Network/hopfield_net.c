#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**************************************CONSTANT DEFINITIONS*******************/
// pixel size
#define PIX 100
// number of patterns
#define NP 3
#define MAXLEN 512

/**************************************GLOBAL VARIABLES*******************/
// pattern matrix
int pattern[NP][PIX];
int pattern_transpose[PIX][NP];
// weight matrix
float weight[PIX][PIX];
// state vector
int state[PIX];
int old_state[PIX];

/**************************************FUNCTION DECLARATIONS*******************/
void display_state(void);
void display_matrix(int rows, int columns, float matrix[][columns]);
void copy_array(int N, int source[N], int des[N]);
int is_equal(int N, int arr1[N], int arr2[N]);
void scaling(int pattern[NP][PIX]);
void scaling_back(int state[PIX]);
void transpose(int pattern[NP][PIX]);
void outer_product(int pattern_transpose[PIX][NP], int pattern[NP][PIX]);
void normalize_weight(void);
int transfer_function(int state[PIX], int i, float net);

/**************************************HOPFIELD NETWORK*******************/
int main(int argc, char *argv[])
{
   if (argc != 3)
   {
        fprintf(stderr, "Usage : %s pattern_file distorted_file \n", argv[0]);
        exit (1);
   }

   FILE *fp1 = NULL;
   FILE *fp2 = NULL;

   // opening pattern_file
   if ((fp1 = fopen(argv[1], "r")) == NULL)
   {
        fprintf(stderr, "error reading %s \n", argv[1]);
        exit (2);
   }
   
   int i = 0, j = 0;
   char line[MAXLEN];
   char *ptr = NULL;

   // storing pattern
   while (fgets(line, MAXLEN, fp1) != NULL)
   {
       ptr = strtok(line, ",");
       pattern[i][j] = atoi(ptr);
       //printf("i %d j %d value %d \n", i, j, pattern[i][j]);
       for (j = 1; j < PIX; j++)
       {
            ptr = strtok(NULL, ",");
            pattern[i][j] = atoi(ptr);
            //printf("i %d j %d value %d \n", i, j, pattern[i][j]);
       }
       j = 0;
       i++;
   }
   fclose(fp1);

   // opening distorted_file
   if ((fp2 = fopen(argv[2], "r")) == NULL)
   {
        fprintf(stderr, "error reading %s \n", argv[2]);
        exit (3);
   }
   i = 0;
   ptr = NULL;
   // initializing state vector
   while (fgets(line, MAXLEN, fp2) != NULL)
   {
        ptr = strtok(line, ",");
        state[i] = atoi(ptr);

        for (i = 1; i < PIX; i++)
	{
             ptr = strtok(NULL, ",");
             state[i] = atoi(ptr);
        }
   }
   fclose(fp2);
   
   scaling(pattern);
   display_state();
   transpose(pattern);
   outer_product(pattern_transpose, pattern);

  // printf("Weight matrix\n");
  // display_matrix(PIX, PIX, weight);
   normalize_weight();
  // printf("Weight matrix after Normalisation\n");
  // display_matrix(PIX, PIX, weight);

   int neuron = 0;
   float net = 0;
   int stable = 1;
   int iteration = 0;
   while (stable != 0)
   {
        j++;
        copy_array(PIX, state, old_state);
        for (neuron = 0; neuron < PIX; neuron++)
        {
             i = 0;
             for (i = 0; i < PIX; i++)
             {
                  net += weight[i][neuron] * (float) state[i];
             }
             state[neuron] = transfer_function(state, neuron, net);
	     net = 0;
             iteration++;
             printf("Iteration : %d\n", iteration);
             display_state();
        }
	stable = is_equal(PIX, state, old_state);
        if (stable == 0)
            break;
        neuron = 0;
   }
   printf("Converged after %d iteration(s)\n", iteration);
   scaling_back(state);
   display_state();
   
   exit (0);
}

/**************************************FUNCTION DEFINITIONS*******************/

void display_state(void)
{
   int i = 0;
   for (i = 0; i < PIX; i++)
   {
       if (state[i] == 1)
           printf("\x1b[31m%d\x1b[0m ", state[i]);
       else if (state[i] == -1) // 0 as -1
           printf("0 ");
       else if (state[i] == 0)
           printf("%d ", state[i]);

       if ((i+1) % 10 == 0)
           printf("\n");
   }
   printf("\n");
   return ;
}

void display_matrix(int rows, int columns, float matrix[][columns])
{
   int i = 0, j = 0;
   for (i = 0; i < rows; i++)
   {
        for (j = 0; j < columns; j++)
        {
             printf("%f ", matrix[i][j]);
	}
	printf("\n");
   }
   return ;
}

void copy_array(int N, int source[N], int des[N])
{
   int i = 0;
   for (i = 0; i < N; i++)
       des[i] = source[i];

   return ;
}

int is_equal(int N, int arr1[N], int arr2[N])
{

   int i = 0;
   for (i = 0; i < N; i++)
   {
        if (arr1[i] != arr2[i])
            return 1;
   }
   return 0;
}

// scales binary inputs to 1 and -1
void scaling(int pattern[NP][PIX])
{
   int i = 0, j = 0;
   for (i = 0; i < NP; i++)
   {
        for (j = 0; j < PIX; j++)
        {
             if (pattern[i][j] == 0)
                 pattern[i][j] = -1;
        }
   }
   return ;
}

void scaling_back(int state[PIX])
{
   int i = 0;
   for (i = 0; i < PIX; i++)
   {
        if (state[i] == -1)
            state[i] = 0;
   }
   return ;
}


// transpose of pattern vector
void transpose(int pattern[NP][PIX])
{
   int i = 0, j = 0;
   for (i = 0; i < PIX; i++)
   {
        for (j = 0; j < NP; j++)
            pattern_transpose[i][j] = pattern[j][i];
   }
   return ;
}

// outer product of two matrices 
void outer_product(int pattern_transpose[PIX][NP], int pattern[NP][PIX])
{
   int i = 0, j = 0, k = 0;
   
   for (i = 0; i < PIX; i++)
   {
        for (j = 0; j < PIX; j++)
        {
             for (k = 0; k < NP; k++)
                 weight[i][j] += pattern_transpose[i][k] * pattern[k][j];
        }
   }
   return ;
}

// normalize and making diagonal entries zero
void normalize_weight(void)
{
   int i = 0, j = 0;
   for (i = 0; i < PIX; i++)
   {
        for (j = 0; j < PIX; j++)
        {
             if (i == j)
                 weight[i][j] = 0;
             else
                 weight[i][j] = weight[i][j] / (float) NP;
	}
   }
}
// transfer function (threshold) to ith neuron
int transfer_function(int state[PIX], int i, float net)
{
   int temp = 0;
   if (net > 0)
       state[i] = 1;
   else if (net < 0)
       state[i] = -1;

   temp = state[i];
   return temp; 
}
