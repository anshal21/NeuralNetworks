/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork_;

import Jama.Matrix;
import java.util.ArrayList;

/**
 *
 * @author anshal
 */
public class NeuralNetwork_ {

    /**
     * @param args the command line arguments
     */
   public static void main(String[] args) {
        // TODO code application logic here
       
        DataLoader dl=new DataLoader();
        ArrayList<Matrix> X=dl.readFeatures("dataSet.txt");
        ArrayList<Matrix> Y=dl.readOutput("output.txt");
        
        NN nn=new NN(X.get(0).getRowDimension()-1,3, Y.get(0).getRowDimension(),3);
        GradientDescent gd=new GradientDescent(0.009,5000, X, Y);
        nn=gd.optimize(nn);
        ArrayList<Matrix> TX=dl.readFeatures("TdataSet.txt");
        ArrayList<Matrix> TY=dl.readOutput("Toutput.txt");
        System.out.println("Test Result");
        int cnt=0;
        
        ArrayList<Matrix> HY=new ArrayList<Matrix>();
        for(int i=0;i<TX.size();i++){
            System.out.print((i+1)+"=> ");
            double er=0;
            Matrix H=nn.forwardPropagation(TX.get(i));
            Matrix P=new Matrix(H.getRowDimension(),H.getColumnDimension());
           // Matrix P=new Matrix(H.getRowDimension(),H.getColumnDimension());
            for(int j=0;j<H.getRowDimension();j++){
                double val=H.get(j,0);
                P.set(j,0,val);
                er=er+Math.abs(val-TY.get(i).get(j,0));
               
                System.out.print(val+" ");
            }
            System.out.println("");
            if(er<0.05){
                cnt++;
            }
            HY.add(P);
            
        }
        PlotCurve pc=new PlotCurve();
        pc.setVisible(true);
        pc.setSize(1200,600);
       
        CurvePanel cp=new CurvePanel(X, Y);
        pc.jPanel1.add(cp);
        CurvePanel cp2=new CurvePanel(TX, HY);
        pc.jPanel2.add(cp2);
        double accuracy=(100.0*cnt)/TY.size();
        System.out.println("Accuracy = "+accuracy);
        pc.accuracy.setText("Accuracy="+accuracy);
        if(accuracy>90)
        nn.saveParameters("Params(x<500).txt");
    }
}
