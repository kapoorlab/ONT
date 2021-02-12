package fileListeners;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

import javax.swing.JComboBox;

import ij.WindowManager;
import net.imglib2.type.numeric.real.FloatType;
import pluginTools.TrainingDataCreator;

	
	public class ChooseTrainingImage implements ActionListener {
		
		
		final TrainingDataCreator parent;
		final JComboBox<String> choice;
		
		
		public ChooseTrainingImage(final TrainingDataCreator parent, final JComboBox<String> choice ) {
			
			
			this.parent = parent;
			this.choice = choice;
			
		}


		@Override
		public void actionPerformed(ActionEvent e) {
			
			String imagename = (String) choice.getSelectedItem();
			
	    	parent.impOrig = WindowManager.getImage(imagename);
	    	if(parent.impOrig!=null) {
	    	parent.inputimage = 
    	    		io.SimplifiedIO.openImage(parent.impOrig.getOriginalFileInfo().directory + parent.impOrig.getOriginalFileInfo().fileName, new FloatType());
	    	parent.imageDirectory = new File(parent.impOrig.getOriginalFileInfo().directory);
	    	parent.imageFilename =  parent.impOrig.getOriginalFileInfo().fileName;
	    	
	    	parent.ManualDots = true;
			parent.MatlabDots = false;
			
			if(parent.overlay!=null)
			parent.overlay.clear();
			if(parent.MatlabOvalRois!=null)
			parent.MatlabOvalRois.clear();
	     	
	    	}
	    	parent.Clickrecorder();	
		
}
	
}
