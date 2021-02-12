package fileListeners;
import java.awt.GridBagConstraints;
import java.awt.Insets;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;

import fileListeners.ChooseTrainingImage;
import loadfile.CovistoOneChFileLoader;
import pluginTools.TrainingDataCreator;




public class ONTManualModeListener implements ItemListener {

	public final TrainingDataCreator parent;
	
	
	
	public ONTManualModeListener( final TrainingDataCreator parent) {
		
		this.parent = parent;
	}
	
	
	
	@Override
	public void itemStateChanged(ItemEvent e) {
		
		if (e.getStateChange() == ItemEvent.SELECTED) {
			
			parent.panelFirst.remove(parent.Panelfile);
			parent.panelFirst.validate();
			parent.panelFirst.repaint();
			
			
			CovistoOneChFileLoader original = new CovistoOneChFileLoader(parent.chooseTrainDatastring, parent.blankimageNames);
			parent.Panelfile = original.SingleChannelOption();
			original.ChooseImage.addActionListener(new ChooseTrainingImage(parent, original.ChooseImage));
			
			
			parent.panelFirst.add(parent.Panelfile, new GridBagConstraints(0, 7, 3, 1, 0.0, 0.0, GridBagConstraints.WEST,
					GridBagConstraints.HORIZONTAL, parent.insets, 0, 0));
			parent.ManualDots = true;
			parent.MatlabDots = false;
			
			parent.Panelclicker.remove(parent.headername);
			parent.Panelclicker.remove(parent.ChooseHeader);
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
