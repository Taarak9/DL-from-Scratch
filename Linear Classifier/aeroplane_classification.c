#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*****************************************************************************/
// CONSTANT DEFINITIONS

// number of patterns
#define N 10 
// max length of line for reading input
#define MAXLEN 32

/*****************************************************************************/
// FUNCTION DECLARATIONS
double sigmoid(double x); 
double random_weight(void);


/*****************************************************************************/
// MAIN PROGRAM STARTS HERE
int main(int argc, char *argv[])
{
   // argument count validation
   if (argc != 5)
   {
        fprintf(stderr, "Usage : %s training_set_file learning_rule max_epoch learning_rate\n", argv[0]);
	fprintf(stderr, "Please enter 1 for Perceptron learning rule\n");
	fprintf(stderr, "Please enter 2 for Gradient descent learning rule\n");
        exit (1);
   }

   FILE *fp = NULL;

   // file opening for reading input
   if ((fp = fopen(argv[1], "r")) == NULL)
   {
        fprintf(stderr, "Error opening file %s.", argv[1]);
        exit (2);
   }

   char line[MAXLEN];

   int i = 0;
   char *ptr = NULL;
   // input vector
   double inp[N][2]; 
   // target vector
   double targ[N] = {0}; 

   // storing the input from file
   printf("-------------------------------------------------------------\n");
   printf("TRAINING DATA\n");
   printf("-------------------------------------------------------------\n");
   printf("MASS\t\tSPEED\t\tTARGET\t\n");
   while (fgets(line, MAXLEN, fp) != NULL)
   {
        ptr = strtok(line, ",");
        inp[i][0] = atof(ptr);
    
        ptr = strtok(NULL, ",");
        inp[i][1] = atof(ptr);

        ptr = strtok(NULL, ",");
        targ[i] = atof(ptr);

        printf("%lf\t%lf\t%lf\n", inp[i][0], inp[i][1], targ[i]);

        i++;	
   }

   // weight vector
   double w[3] = {0};
   // bias_weight
   w[0] = random_weight(); 
   // mass_weight
   w[1] = random_weight(); 
   // speed_weight
   w[2] = random_weight(); 
   
   int epoch = 0;
   int final_epoch = 0;

   // learning rate
   double eta = atoi(argv[4]);

   // temp variables
   double C = 0;
   double x = 0;
   double y = 0;
   double z = 0;

   // option
   int op = atoi(argv[2]);

   // output vector
   double out[N] = {0}; 
   
   // maximum number of epochs
   int max_epoch = atoi(argv[3]);
   
   // error vector with index as epoch
   double error[max_epoch];

   // initialization
   i = 0;
   for (i = 0; i < max_epoch; i++)
       error[max_epoch] = 0;


   switch (op)
   {
      case 1 :  // Perceptron learning rule and online learning
                i = 0;
                while (epoch < max_epoch)
                {
                     while (i < N)
                     {
                          out[i] = sigmoid((inp[i][0]*w[1]) + (inp[i][1]*w[2]) - w[0]);
                            
                          w[0] = w[0] - eta * (targ[i] - out[i]);
                          w[1] = w[1] + eta * (targ[i] - out[i]) * inp[i][0];
                          w[2] = w[2] + eta * (targ[i] - out[i]) * inp[i][1];
			    
                          // error
                          C += ( targ[i] - out[i] ) * ( targ[i] - out[i] );
                          i++;
                          printf("%d\t%lf\t%lf\t\n", i, out[i], targ[i]);
                     }
                     error[epoch] = (0.5 * C);
                     epoch++;
                     C = 0;
                     i = 0;
                }
                printf("-------------------------------------------------------------\n");
                printf("PATTERN\tOUTPUT\t\tTARGET\t\n");
                for (i = 0; i < N; i++)
                    printf("%d\t%lf\t%lf\t\n", i, out[i], targ[i]);
                final_epoch = epoch-1;
                break;
          
      case 2 :  // Gradient Descent learning rule using Cross Entropy erro fn
                // batch learning
                i = 0; 
                while (epoch < max_epoch)
                {
                     while (i < N)
                     {
                          out[i] = sigmoid(inp[i][0]*w[1] + inp[i][1]*w[2] - w[0]);

                          // derivate of CE function cancels out sigmoid derivate
                          x -= eta*(targ[i] - out[i]);
                          y += eta*(targ[i] - out[i])*inp[i][0];
                          z += eta*(targ[i] - out[i])*inp[i][1];
			    
                          // CE error for all patterns
                          C += -((targ[i]*log(out[i]) + ((double) 1 - targ[i])*log((double) 1 - out[i])));

                          i++; 
                     }
                     w[0] += x;
                     w[1] += y;
                     w[2] += z;
                       
                     error[epoch] = C / N;
                     epoch++;
                     C = 0;
                     i = 0;
                     x = 0;
                     y = 0;
                     z = 0;
                  }
                  printf("-------------------------------------------------------------\n");
                  printf("PATTERN\tOUTPUT\t\tTARGET\t\n");
                  for (i = 0; i < N; i++)
                      printf("%d\t%lf\t%lf\t\n", i, out[i], targ[i]);
                  final_epoch = epoch-1;
                  break;

        default :  fprintf(stderr, "Please select a valid learning rule\n");
                   exit (3);
   }
   i = 0;


   printf("-------------------------------------------------------------\n");
   printf("ERROR\t\tEPOCH\t\n");
   printf("%lf\t%d\t\n", error[final_epoch], final_epoch);

   /**************************************************************************/
   // decison boundary equation
   // inp[i][1] = -w[1]*inp[i][1]/w[2] + w[0]/w[2]
   // slope
   double m = (double) -1 * (w[1]/w[2]);
   // intercept
   double c = w[0]/w[2];

   // for appending slope and intercept into dataset.csv
   FILE *zfp = NULL;
   char zfile[MAXLEN] = "dataset.csv";

   if ((zfp = fopen(zfile, "a")) == NULL)
   {
        fprintf(stderr, "Error opening dataset.csv file\n");
        exit (4);
   }

   fprintf(zfp, "%lf, %lf, -1\n", m , c);
   fclose(zfp);

   /**************************************************************************/
   // storing coordinates in a file for graph plotting
   FILE *xfp = NULL;
   FILE *yfp = NULL;
   char xfile[MAXLEN] = "xfile";
   char yfile[MAXLEN] = "yfile";

   if ((xfp = fopen(xfile, "w")) == NULL)
   {
        fprintf(stderr, "Error opening x file\n");
        exit (4);
   }

   if ((yfp = fopen(yfile, "w")) == NULL)
   {
        fprintf(stderr, "Error opening y file\n");
        exit (5);
   }

   i = 0;
   for (i = 0; i < final_epoch; i++)
   {
        fprintf(yfp, "%f\n", error[i]);
	fprintf(xfp, "%d\n", i);
   }

   fclose(xfp);
   fclose(yfp);

   exit (0);
}

/*****************************************************************************/
// transfer function
double sigmoid(double x)
{
   double y = 0;
   y = (double) 1 / ((double) 1 + exp(-x));

   return y;
}

// random weight between -3 and 3
double random_weight(void)
{
   double rand_num = 0;

   rand_num = (double) rand() / (double) RAND_MAX;

   rand_num = (rand_num * (double) 6) - (double) 3;

   return rand_num;
}

