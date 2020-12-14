package fileListeners;

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

import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.OvalRoi;
import net.imglib2.RealPoint;
import net.imglib2.type.numeric.real.FloatType;
import pluginTools.TrainingDataCreator;
import pointSelector.Roiobject;

public class LoadCSV implements ActionListener {

	final TrainingDataCreator parent;
	final JComboBox<String> choice;
	boolean isDone = false;

	public LoadCSV(final TrainingDataCreator parent, JComboBox<String> choice) {

		this.parent = parent;
		this.choice = choice;

	}

	@Override
	public void actionPerformed(ActionEvent e) {

		// Choose Image
		String imagename = (String) choice.getSelectedItem();

		parent.impOrig = WindowManager.getImage(imagename);
		if (parent.impOrig != null) {
			ImagePlus.removeImageListener(parent.Ivl);
			parent.inputimage = io.SimplifiedIO.openImage(
					parent.impOrig.getOriginalFileInfo().directory + parent.impOrig.getOriginalFileInfo().fileName,
					new FloatType());
			parent.imageDirectory = new File(parent.impOrig.getOriginalFileInfo().directory);
			parent.imageFilename = parent.impOrig.getOriginalFileInfo().fileName;

		
			JFileChooser csvfile = new JFileChooser();
			FileFilter csvfilter = new FileFilter() {
				// Override accept method
				public boolean accept(File file) {

					// if the file extension is .log return true, else false
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
			if (parent.impOrig != null)
				csvfile.setCurrentDirectory(new File(parent.impOrig.getOriginalFileInfo().directory));
			else
				csvfile.setCurrentDirectory(new java.io.File("."));
			csvfile.setDialogTitle("Matlab Time-Location File");
			csvfile.setFileSelectionMode(JFileChooser.FILES_ONLY);
			csvfile.setFileFilter(csvfilter);
			int count = 0;
			if (parent.overlay != null)
				parent.overlay.clear();
			if (parent.MatlabOvalRois != null)
				parent.MatlabOvalRois.clear();
			if (csvfile.showOpenDialog(parent.Cardframe) == JFileChooser.APPROVE_OPTION) {
				parent.Matlabfile = new File(csvfile.getSelectedFile().getPath());
				ArrayList<Roiobject> Allrois = new ArrayList<Roiobject>();

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

				} catch (IOException f) {
					f.printStackTrace();
				}

				parent.ManualDots = false;
				parent.MatlabDots = true;
				parent.addToName = csvfile.getSelectedFile().getName().replaceFirst("[.][^.]+$", "");
				parent.eventfieldname.setText(parent.addToName);
				parent.impOrig.updateAndDraw();
				parent.Clickrecorder();
			} else
				csvfile = null;
		}

	}
}