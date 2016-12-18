/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork_;
import Jama.*;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Writer;
import java.util.logging.Level;
import java.util.logging.Logger;
/**
 *
 * @author anshal dwivedi
 * B.tech 3rdYear MNNIT Allahabad
 * anshaldwivedi@gmail.com
 * This is Neural Network Based on Andrew Ng's Coursera Course :)
 */
public class NN {
  int FEATURES,HIDDEN_LAYERS;
  int OUTPUT_CLASS;
  int HIDDEN_UNITS;
  Matrix Activation[];
  Matrix Z[];
  Matrix theta[];
  Matrix delta[],DELTA[];
  Matrix Derivative[];
  int s[];
  double error=0;
  double LAMBDA=0.1;
  public NN(int f,int h,int o,int hu){
      FEATURES=f;
      HIDDEN_LAYERS=h;
      OUTPUT_CLASS=o;
      HIDDEN_UNITS=hu;
      Matrix X=new Matrix(FEATURES,1);
      s=new int[HIDDEN_LAYERS+2];
      s[0]=FEATURES;
      Activation=new Matrix[HIDDEN_LAYERS+2];
      Z=new Matrix[HIDDEN_LAYERS+2];
      theta=new Matrix[HIDDEN_LAYERS+1];
      DELTA=new Matrix[HIDDEN_LAYERS+1];
      Derivative=new Matrix[HIDDEN_LAYERS+1];
      s[HIDDEN_LAYERS+1]=OUTPUT_CLASS;
      for(int i=1;i<=HIDDEN_LAYERS;i++)
          s[i]=HIDDEN_UNITS;
      delta=new Matrix[HIDDEN_LAYERS+2];
      for(int i=0;i<HIDDEN_LAYERS+1;i++){
          theta[i]=new Matrix(s[i+1],s[i]+1,1);
          DELTA[i]=new Matrix(s[i+1],s[i]+1,1);
          Derivative[i]=new Matrix(s[i+1],s[i]+1,1);
      }
      for(int i=0;i<HIDDEN_LAYERS+1;i++){
          theta[i]=fillRandom(theta[i]);
      }
      for(int i=1;i<HIDDEN_LAYERS+2;i++){
          Z[i]=new Matrix(s[i],1);
          Activation[i]=new Matrix(s[i],1);
      }
  }
  public double sigmoid(double z){              //Calculation of Sigmoid function
        return 1.0/(1+Math.exp(-(z)));
  }
  public Matrix fillRandom(Matrix A){           //Random intitialization of a matrix
      for(int i=0;i<A.getRowDimension();i++){
          for(int j=0;j<A.getColumnDimension();j++){
              A.set(i,j,Math.random()/100);
          }
      }
      return A;
  }
  public Matrix g(Matrix a){                    //Vectorized Implementation of Sigmoid Function 
      Matrix b=new Matrix(a.getRowDimension()+1,a.getColumnDimension());
      b.set(0,0,1);
      double mean=sum(a)/(a.getColumnDimension()*a.getRowDimension());
      for(int i=0;i<a.getRowDimension();i++){
          for(int j=0;j<a.getColumnDimension();j++){
              b.set(i+1, j,sigmoid(a.get(i,j)));
          }
      }
      return b;
  }
  public Matrix forwardPropagation(Matrix X){       //Forward Propagation for Hypothesis Calculation
     // printStartLine();
      Activation[0]=X.copy();
      //Activation[0].print(1,3);
      
      for(int l=1;l<HIDDEN_LAYERS+2;l++){
          Z[l]=theta[l-1].times(Activation[l-1]);  
          Activation[l]=g(Z[l]);                    //Activation value calculation
         // Activation[l].print(1,3);
          //System.out.println("with sum = "+sum(Activation[l]));
      }
     // printEndLine();
      Matrix H=new Matrix(Activation[HIDDEN_LAYERS+1].getRowDimension()-1,1);
      for(int i=1;i<Activation[HIDDEN_LAYERS+1].getRowDimension();i++)    //Removing Biased unit from the final results
          H.set(i-1,0,Activation[HIDDEN_LAYERS+1].get(i,0));
       //   H.print(1,5);
      return H;
  }
  public Matrix mult(Matrix A,Matrix B){        //ElementWise Matrix Multiplication
      Matrix C=new Matrix(A.getRowDimension(),A.getColumnDimension());
      for(int i=0;i<A.getRowDimension();i++){
          for(int j=0;j<A.getColumnDimension();j++){
              C.set(i,j,A.get(i,j)*B.get(i,j));
          }
      }
      return C;
  }
   public double sum(Matrix A){    //Gives Sum of element of A matrix
          double sum=0;
          for(int i=0;i<A.getRowDimension();i++){
              for(int j=0;j<A.getColumnDimension();j++){
                  sum+=A.get(i,j);
              }
          }
          return sum;
      }
    public double max(Matrix A){    //Find max element in a matrix;
          double max=0;
          for(int i=0;i<A.getRowDimension();i++){
              for(int j=0;j<A.getColumnDimension();j++){
                  max=Math.max(max,A.get(i,j));
              }
          }
          return max;
      }
  public Matrix oneMinus(Matrix A){         //caluclate 1-A where A is Matrix
      for(int i=0;i<A.getRowDimension();i++){
          for(int j=0;j<A.getColumnDimension();j++){
              A.set(i,j,1-A.get(i,j));
          }
      }
      return A;
  }
  public void backPropagation(Matrix X,Matrix Y){  //Application of back Propagation to calculate error "delta" .
      //Matrix Y=new Matrix(OUTPUT_CLASS,1,0);
      //Y.set(1,0,1);
      for(int i=1;i<=HIDDEN_LAYERS+1;i++){
          delta[i]=new Matrix(s[i],1);
      }
     
      Matrix H=forwardPropagation(X);
//      H.print(1,3);
//      Y.print(1,3);
      delta[HIDDEN_LAYERS+1]=H.minus(Y);    //error for the last layer is direct differnence from correct result
     // delta[HIDDEN_LAYERS+1].print(1,3);
      for(int l=HIDDEN_LAYERS;l>=1;l--){
          Matrix thetaDup=new Matrix(theta[l].getRowDimension(),theta[l].getColumnDimension()-1);
          for(int i=0;i<theta[l].getRowDimension();i++){
              for(int j=1;j<theta[l].getColumnDimension();j++){
                  thetaDup.set(i,j-1,theta[l].get(i,j));
              }
          }
          Matrix a=new Matrix(Activation[l].getRowDimension()-1,1);
          for(int i=1;i<Activation[l].getRowDimension();i++){
              a.set(i-1,0,Activation[l].get(i,0));
          }
//          thetaDup.print(10,10);
//          delta[l+1].print(1,10);
          Matrix p1=(thetaDup.transpose()).times(delta[l+1]);
//          p1.print(1,10);
          Matrix p2=mult(a,oneMinus(a));
          delta[l]=mult(p1,p2);
          error+=sum(delta[l]);
      }
//      printStartLine();
//      for(int l=HIDDEN_LAYERS+1;l>=1;l--){
//          delta[l].print(delta[l].getRowDimension(),12);
//      }
//      printEndLine();
      
  }
  
  public void updateDelta(){
      for(int l=0;l<HIDDEN_LAYERS+1;l++){
          for(int i=0;i<theta[l].getRowDimension();i++){
              for(int j=0;j<theta[l].getColumnDimension();j++){
                  double val=Activation[l].get(j,0)*delta[l+1].get(i,0); // DELTA[L][I][J]+=A[L][J]*delta[L+1][I]
                  DELTA[l].set(i,j,DELTA[l].get(i,j)+val);
              }
          }
      }
  }
  public void printStartLine(){
      for(int i=0;i<100;i++){
          System.out.print("-");
      }
      System.out.println("");
  }
    public void printEndLine(){
      for(int i=0;i<100;i++){
          System.out.print("=");
      }
      System.out.println("");
  }
  public void calculateDerivative(int m){       //Caluclation of D[l][i][j]=DELTA[l][i][j]/m+lambda*theta[l][i][j]
      for(int l=0;l<HIDDEN_LAYERS+1;l++){
          for(int i=0;i<theta[l].getRowDimension();i++){
              for(int j=0;j<theta[l].getColumnDimension();j++){
                  if(j!=0){
                      double val=DELTA[l].get(i,j)/m+LAMBDA*theta[l].get(i,j);
                      Derivative[l].set(i,j,val); 
                  }
                  else{
                      double val=DELTA[l].get(i,j)/m;
                      Derivative[l].set(i,j,val);
                  }
              }
          }
      }
  }
  //Upadation of Parameters according to Gradient Descent Algo , change the definition according to
  //your optimization Algo.
  public void updateTheta(double ALPHA){   
      for(int l=0;l<=HIDDEN_LAYERS;l++){
          for(int i=0;i<theta[l].getRowDimension();i++){
              for(int j=0;j<theta[l].getColumnDimension();j++){
                  theta[l].set(i,j,theta[l].get(i,j)-ALPHA*Derivative[l].get(i,j));
              }
          }
      }
  }
  void print(Matrix A[]){
      for(int l=0;l<A.length;l++){
          for(int i=0;i<A[l].getRowDimension();i++){
              for(int j=0;j<A[l].getColumnDimension();j++){
                  System.out.print(A[l].get(i,j)+" ");
              }
              System.out.println("");
          }
      }
  }
  // Save Calculated value of Weights in file
  void saveParameters(String Name){
      try {
          File f=new File(Name);
          if(f.exists())
              f.delete();
          f.createNewFile();
         
          PrintWriter pw=new PrintWriter(f);
          for(int i=0;i<HIDDEN_LAYERS+1;i++){
              theta[i].print(pw, 1, 10);
          }
          pw.close();
      } catch (IOException ex) {
          Logger.getLogger(NN.class.getName()).log(Level.SEVERE, null, ex);
      }
      
  }
}
