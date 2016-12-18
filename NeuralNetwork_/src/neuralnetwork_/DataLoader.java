/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork_;

import Jama.Matrix;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author anshal
 */
public class DataLoader {
    public ArrayList<Matrix> readFeatures(String name){
        try {
            Scanner sc=new Scanner(new File(name));
            ArrayList<Matrix> am=new ArrayList<Matrix>();
            while(sc.hasNextLine()){
                String s=sc.nextLine();
                String[] dd=s.split("\\s+");
                double[][] data=new double[dd.length][1];
                for(int i=0;i<dd.length;i++){
                    data[i][0]=Double.parseDouble(dd[i]);
                }
                Matrix A=new Matrix(data);
                am.add(A);
            }
            return am;  
        } catch (FileNotFoundException ex) {
            Logger.getLogger(DataLoader.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }
    
    public ArrayList<Matrix> readOutput(String name){
        try {
            Scanner sc=new Scanner(new File(name));
            ArrayList<Matrix> am=new ArrayList<Matrix>();
            while(sc.hasNextLine()){
                String s=sc.nextLine();
                String[] dd=s.split("\\s+");
                double[][] data=new double[dd.length][1];
                for(int i=0;i<dd.length;i++){
                    data[i][0]=Double.parseDouble(dd[i]);
                }
                Matrix A=new Matrix(data);
                am.add(A);
            }
            return am;  
        } catch (FileNotFoundException ex) {
            Logger.getLogger(DataLoader.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }
}
