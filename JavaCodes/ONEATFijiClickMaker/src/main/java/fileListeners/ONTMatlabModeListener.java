package fileListeners;
import java.awt.GridBagConstraints;
import java.awt.Insets;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;

import ONEATloadfile.CovistoOneChFileLoader;
import ONEATloadfile.CovistoTwoChDropbox;
import ONEATloadfile.CovistoTwoChForceFileLoader;
import fileListeners.ChooseTrainingImage;
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
			
			
		 	ActionListener newAction = new LoadCSV( parent, originalncsv.ChooseImage );
	    	// Choose CSV
	    	
		 	originalncsv.ChooseFile.addActionListener(newAction);
			
			
			parent.panelFirst.add(parent.Panelfile, new GridBagConstraints(0, 7, 3, 1, 0.0, 0.0, GridBagConstraints.WEST,
					GridBagConstraints.HORIZONTAL, parent.insets, 0, 0));
			parent.Panelclicker.add(parent.headername, new GridBagConstraints(3, 0, 3, 1, 0.0, 0.0, GridBagConstraints.WEST,
					GridBagConstraints.HORIZONTAL, new Insets(10, 10, 0, 10), 0, 0));
			parent.Panelclicker.add(parent.ChooseHeader, new GridBagConstraints(3, 1, 3, 1, 0.0, 0.0, GridBagConstraints.WEST,
					GridBagConstraints.HORIZONTAL, new Insets(10, 10, 0, 10), 0, 0));
		
			parent.Panelclicker.validate();
			parent.Panelclicker.repaint();
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
