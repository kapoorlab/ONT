package fileListeners;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import javax.swing.JComboBox;

import ONEATloadfile.CovistoTwoChDropbox;
import ij.WindowManager;
import ij.gui.OvalRoi;
import net.imglib2.RealPoint;
import net.imglib2.type.numeric.real.FloatType;
import pluginTools.TrainingDataCreator;
import pointSelector.Roiobject;

public class ONTHeaderListener implements ActionListener {
		
		
		final TrainingDataCreator parent;
		final JComboBox<String> choice;
		
		
		public ONTHeaderListener(final TrainingDataCreator parent, final JComboBox<String> choice ) {
			
			
			this.parent = parent;
			this.choice = choice;
			
		}


		@Override
		public void actionPerformed(ActionEvent e) {
			
			String headername = (String) choice.getSelectedItem();
			
	    	parent.header = headername;
	    	if (parent.overlay != null)
				parent.overlay.clear();
			if (parent.MatlabOvalRois != null)
				parent.MatlabOvalRois.clear();
	    	ArrayList<Roiobject> Allrois = new ArrayList<Roiobject>();
	    	String line = "";
			String cvsSplitBy = ",";
			int count = 0;
			try (BufferedReader br = new BufferedReader(new FileReader(parent.Matlabfile))) {
	    	while ((line = br.readLine()) != null) {

				// use comma as separator
				// Csv file has structure T Y X Angle
				String[] TYXApoints = line.split(cvsSplitBy);
				
				if (count > 0) {

					int Y, X;
					if(TYXApoints[0]!="NAN" || TYXApoints[1]!="NAN" || TYXApoints[2]!="NAN") {
					int T = (int) Float.parseFloat(TYXApoints[0]);
					if (parent.header == "Y") {
						try {
						Y = (int) Float.parseFloat(TYXApoints[1]);
						X = (int) Float.parseFloat(TYXApoints[2]);
						
						int Angle = 2;
						if (TYXApoints.length > 3)
							Angle = (int) Float.parseFloat(TYXApoints[3]);

						if (parent.MatlabOvalRois.get(T) == null) {
							Allrois = new ArrayList<Roiobject>();
							parent.MatlabOvalRois.put(T, Allrois);
						} else
							parent.MatlabOvalRois.put(T, Allrois);
						if (X > 0 && Y > 0 && X < parent.inputimage.dimension(0) && Y < parent.inputimage.dimension(1) ) {
						
						OvalRoi roi = new OvalRoi(X, Y, 10, 10);
						Allrois.add(new Roiobject(parent.RejectColor, roi,
								new RealPoint(new double[] { X, Y, Angle })));
						
						}
						
						}
						catch(NumberFormatException Nan) {
							
							
						}
					} else {
						try {
						Y = (int) Float.parseFloat(TYXApoints[2]);
						X = (int) Float.parseFloat(TYXApoints[1]);
						
						int Angle = 2;
						if (TYXApoints.length > 3)
							Angle = (int) Float.parseFloat(TYXApoints[3]);

						if (parent.MatlabOvalRois.get(T) == null) {
							Allrois = new ArrayList<Roiobject>();
							parent.MatlabOvalRois.put(T, Allrois);
						} else
							parent.MatlabOvalRois.put(T, Allrois);
						if (X > 0 && Y > 0 && X < parent.inputimage.dimension(0) && Y < parent.inputimage.dimension(1) ) {
						
						OvalRoi roi = new OvalRoi(X, Y, 10, 10);
						Allrois.add(new Roiobject(parent.RejectColor, roi,
								new RealPoint(new double[] { X, Y, Angle })));
						
						}
						}
						catch(NumberFormatException Nan) {
							
							
						}
					}
			

				}
				}
				count = count + 1;
			}
	    	
			} catch (FileNotFoundException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			
			
			parent.impOrig.updateAndDraw();
}

}
