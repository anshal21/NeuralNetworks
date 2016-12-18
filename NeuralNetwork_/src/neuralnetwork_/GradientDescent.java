package neuralnetwork_;

import Jama.Matrix;
import java.util.ArrayList;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author anshal dwivedi
 * This is an implementation of Optimization Algorithm Gradient Descent;
 */
public class GradientDescent {
    public int ITERATION;
    double ALPHA;
    ArrayList<Matrix> X,Y;
    public GradientDescent(double ALPHA,int ITERATION,ArrayList<Matrix> X,ArrayList<Matrix> Y){
        this.ALPHA=ALPHA;
        this.ITERATION=ITERATION;
        this.X=X;
        this.Y=Y;
    }
    
    public NN optimize(NN nn){
    //    NN nn=new NN(X.get(0).getRowDimension()-1,2, Y.get(0).getRowDimension(),5);
        nn.print(nn.theta);
        double prev=10000;
        for(int itr=0;itr<ITERATION;itr++){
           nn.error=0;
            for(int i=0;i<X.size();i++){
                nn.backPropagation(X.get(i),Y.get(i));
                nn.updateDelta();
            }
            nn.calculateDerivative(X.size());
            nn.updateTheta(ALPHA);
             System.out.println("ITERATION : "+itr+" with error = "+nn.error);
            double er=Math.abs(nn.error);
//            if(er>prev){
//                break;
//            }
            prev=er;
        }
        nn.print(nn.theta);
        
        return nn;
    }
}
