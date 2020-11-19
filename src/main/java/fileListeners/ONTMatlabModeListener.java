package fileListeners;
import java.awt.GridBagConstraints;
import java.awt.Insets;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;

import fileListeners.ChooseTrainingImage;
import loadfile.CovistoOneChFileLoader;
import loadfile.CovistoTwoChForceFileLoader;
import pluginTools.TrainingDataFileChooser;




public class ONTMatlabModeListener implements ItemListener {

	public final TrainingDataFileChooser parent;
	
	
	
	public ONTMatlabModeListener( final TrainingDataFileChooser parent) {
		
		this.parent = parent;
	}
	
	
	
	@Override
	public void itemStateChanged(ItemEvent e) {
		
		if (e.getStateChange() == ItemEvent.SELECTED) {
			
			parent.panelFirst.remove(parent.Panelfile);
			parent.panelFirst.validate();
			parent.panelFirst.repaint();
			
			CovistoTwoChForceFileLoader originalncsv = new CovistoTwoChForceFileLoader(parent.chooseMatlabTrainDatastring, parent.blankimageNames);
			parent.Panelfile = originalncsv.TwoChannelOption();
			
			originalncsv.ChooseImage.addActionListener(new ChooseTrainingImageMatlabcsv(parent, originalncsv.ChooseImage, originalncsv.ChoosesecImage));
			
			
			parent.panelFirst.add(parent.Panelfile, new GridBagConstraints(0, 7, 3, 1, 0.0, 0.0, GridBagConstraints.WEST,
					GridBagConstraints.HORIZONTAL, parent.insets, 0, 0));
			
			parent.ManualDots = false;
			parent.MatlabDots = true;
		parent.Panelfile.validate();
		parent.Panelfile.repaint();
		
		parent.panelFirst.validate();
		parent.panelFirst.repaint();
		parent.Cardframe.pack();
		}
		
		else if (e.getStateChange() == ItemEvent.DESELECTED) {
			
	
			
		}
		
		
		
		
	}

}
