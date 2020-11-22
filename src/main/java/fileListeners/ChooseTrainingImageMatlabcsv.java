package fileListeners;

import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileFilter;

import ij.WindowManager;
import ij.gui.OvalRoi;
import net.imglib2.RealPoint;
import net.imglib2.type.numeric.real.FloatType;
import pluginTools.TrainingDataCreator;
import pointSelector.Roiobject;

public class ChooseTrainingImageMatlabcsv implements ActionListener {
	
	
	final TrainingDataCreator parent;
	final JComboBox<String> choice;
	final JComboBox<String> choicecsv;
	
	
	public ChooseTrainingImageMatlabcsv(final TrainingDataCreator parent, final JComboBox<String> choice, final JComboBox<String> choicecsv ) {
		
		
		this.parent = parent;
		this.choice = choice;
		this.choicecsv = choicecsv;
		
	}


	@Override
	public void actionPerformed(ActionEvent e) {
		

		// Choose Image
		String imagename = (String) choice.getSelectedItem();
		
    	parent.impOrig = WindowManager.getImage(imagename);
    	parent.inputimage = 
	    		io.SimplifiedIO.openImage(parent.impOrig.getOriginalFileInfo().directory + parent.impOrig.getOriginalFileInfo().fileName, new FloatType());
    	parent.imageDirectory = new File(parent.impOrig.getOriginalFileInfo().directory);
    	parent.imageFilename =  parent.impOrig.getOriginalFileInfo().fileName;
    	
    	// Choose CSV
    	String csvname = (String) choicecsv.getSelectedItem();
    	
    	File Matlabfile = new File(csvname);
    	
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
		
		
		final Boolean isCSV = csvfilter.accept(Matlabfile);
		
		
        String cvsSplitBy = ",";
		int count = 0;
		String line = "";
		if (isCSV) {
			
			ArrayList<Roiobject> Allrois = new ArrayList<Roiobject>();
			
	        try (BufferedReader br = new BufferedReader(new FileReader(Matlabfile))) {

	            while ((line = br.readLine()) != null) {

	                // use comma as separator
	            	// Csv file has structure T Y X Angle
	                String[] TYXApoints = line.split(cvsSplitBy);
                      
	                 if(count > 0) {
	                	 
	                   
	                   int T = Integer.parseInt(TYXApoints[0]);
	                   int Y = Integer.parseInt(TYXApoints[1]);
	                   int X = Integer.parseInt(TYXApoints[2]);
	                   int Angle = 0;
	                   if(TYXApoints.length > 3)
	                	   Angle = Integer.parseInt(TYXApoints[3]);
	                	 
                       if(parent.MatlabOvalRois.get(T)==null) {
                    	    Allrois = new ArrayList<Roiobject>();
                    	    parent.MatlabOvalRois.put(T, Allrois);    
                       }
                       else
                    	   parent.MatlabOvalRois.put(T, Allrois);
	                OvalRoi roi = new OvalRoi(X, Y, 10, 10);
	                
	                Allrois.add(new Roiobject (Color.RED, roi, 
	                		new RealPoint(new double[] {X, Y, Angle})));
	         
	            }
	                 count = count +  1;
	            }
	            
	            
	        } catch (IOException f) {
	            f.printStackTrace();
	        }
	        
	        
		}
		else
			Matlabfile = null;
    	
    	
}
}