package fileListeners;
import java.awt.GridBagConstraints;
import java.awt.Insets;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;

import fileListeners.ChooseTrainingImage;
import loadfile.CovistoOneChFileLoader;
import loadfile.CovistoTwoChDropbox;
import loadfile.CovistoTwoChForceFileLoader;
import pluginTools.TrainingDataCreator;




public class ONTMatlabModeListener implements ItemListener {

	public final TrainingDataCreator parent;
	
	
	
	public ONTMatlabModeListener( final TrainingDataCreator parent) {
		
		this.parent = parent;
	}
	
	
	
	@Override
	public void itemStateChanged(ItemEvent e) {
		
		if (e.getStateChange() == ItemEvent.SELECTED) {
			
			parent.panelFirst.remove(parent.Panelfile);
			parent.panelFirst.validate();
			parent.panelFirst.repaint();
			
			CovistoTwoChDropbox originalncsv = new CovistoTwoChDropbox(parent.chooseMatlabTrainDatastring, parent.blankimageNames);
			parent.Panelfile = originalncsv.TwoChannelOption();
			
			originalncsv.ChooseImage.addActionListener(new ChooseTrainingImageMatlabcsv(parent, originalncsv.ChooseImage, originalncsv.ChooseFile));
			
			
			parent.panelFirst.add(parent.Panelfile, new GridBagConstraints(0, 7, 3, 1, 0.0, 0.0, GridBagConstraints.WEST,
					GridBagConstraints.HORIZONTAL, parent.insets, 0, 0));
			
		
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
