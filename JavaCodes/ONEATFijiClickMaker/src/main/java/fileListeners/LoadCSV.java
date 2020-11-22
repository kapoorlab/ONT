package fileListeners;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import javax.swing.JFileChooser;
import javax.swing.filechooser.FileFilter;

import ij.gui.OvalRoi;
import net.imglib2.RealPoint;
import pluginTools.TrainingDataCreator;
import pointSelector.Roiobject;

public class LoadCSV implements ActionListener {
	
	final TrainingDataCreator parent;
	
	public LoadCSV(final TrainingDataCreator parent) {
	
		this.parent = parent;
		
	}
	@Override
	public void actionPerformed(ActionEvent e) {
	
		JFileChooser csvfile = new JFileChooser();
		FileFilter csvfilter = new FileFilter() 
		{
		      //Override accept method
		      public boolean accept(File file) {
		              
		             //if the file extension is .log return true, else false
		             if (file.getName().endsWith(".csv")) {
		                return true;
		             }
		             return false;
		      }

			@Override
			public String getDescription() {
				
				return null;
			}
		};
        String line = "";
        String cvsSplitBy = ",";
		if (parent.impOrig!=null)
		csvfile.setCurrentDirectory(new File(parent.impOrig.getOriginalFileInfo().directory));
		else 
			csvfile.setCurrentDirectory(new java.io.File("."));
		csvfile.setDialogTitle("Green Cell CSV file");
		csvfile.setFileSelectionMode(JFileChooser.FILES_ONLY);
		csvfile.setFileFilter(csvfilter);
		int count = 0;
		
		if (csvfile.showOpenDialog(parent.Cardframe) == JFileChooser.APPROVE_OPTION) {
			File Matlabfile = new File(csvfile.getSelectedFile().getPath());
			ArrayList<Roiobject> Allrois = new ArrayList<Roiobject>(); 
		
        try (BufferedReader br = new BufferedReader(new FileReader(Matlabfile))) {

            while ((line = br.readLine()) != null) {

                // use comma as separator
            	// Csv file has structure T Y X Angle
                String[] TYXApoints = line.split(cvsSplitBy);
                  String FirstHead = "", SecondHead = "", ThirdHead="";
                  
                 if(count == 0) {
                	 
                	 FirstHead = TYXApoints[0];
                	 SecondHead = TYXApoints[1];
                	 ThirdHead = TYXApoints[2];
                 }
                
                 if(count > 0) {
                	 
                   int Y, X;
                   int T = (int)Float.parseFloat(TYXApoints[0]);
                   if(SecondHead == "Y") {
                      Y = (int)Float.parseFloat(TYXApoints[1]);
                      X = (int)Float.parseFloat(TYXApoints[2]);  
                   }
                   else {
                	   Y = (int)Float.parseFloat(TYXApoints[2]);
                	   X = (int)Float.parseFloat(TYXApoints[1]);
                   }
                   int Angle = 0;
                   if(TYXApoints.length > 3)
                	   Angle = (int)Float.parseFloat(TYXApoints[3]);
                	 
                   if(parent.MatlabOvalRois.get(T)==null) {
                	    Allrois = new ArrayList<Roiobject>();
                	    parent.MatlabOvalRois.put(T, Allrois);    
                   }
                   else
                	   parent.MatlabOvalRois.put(T, Allrois);
                OvalRoi roi = new OvalRoi(X, Y, 10, 10);
                Allrois.add(new Roiobject (parent.RejectColor, roi, 
                		new RealPoint(new double[] {X, Y, Angle})));
         
            }
                 count = count +  1;
            }
            
            
        } catch (IOException f) {
            f.printStackTrace();
        }
        
        parent.ManualDots = false;
		parent.MatlabDots = true;
		
		
     	parent.Clickrecorder();	
	}
	else
		csvfile = null;
	
	
}
}