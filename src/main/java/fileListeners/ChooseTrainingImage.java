package fileListeners;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JComboBox;

import ij.WindowManager;
import pluginTools.TrainingDataFileChooser;

	
	public class ChooseTrainingImage implements ActionListener {
		
		
		final TrainingDataFileChooser parent;
		final JComboBox<String> choice;
		
		
		public ChooseTrainingImage(final TrainingDataFileChooser parent, final JComboBox<String> choice ) {
			
			
			this.parent = parent;
			this.choice = choice;
			
		}


		@Override
		public void actionPerformed(ActionEvent e) {
			
			String imagename = (String) choice.getSelectedItem();
			
	    	parent.impOrig = WindowManager.getImage(imagename);

}
	
}
