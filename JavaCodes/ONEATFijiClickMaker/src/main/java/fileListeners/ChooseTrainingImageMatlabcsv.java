package fileListeners;

import java.awt.Color;
import java.awt.TextField;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import javax.swing.JButton;
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
	final JButton choicecsv;
	
	
	public ChooseTrainingImageMatlabcsv(final TrainingDataCreator parent, final JComboBox<String> choice, final JButton choicecsv ) {
		
		
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
    	choicecsv.addActionListener(new LoadCSV(parent));
    	
    	
    	
	}
	
}
	
	