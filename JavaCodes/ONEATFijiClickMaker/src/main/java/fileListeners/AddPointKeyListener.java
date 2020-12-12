package fileListeners;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import pluginTools.TrainingDataCreator;

public class AddPointKeyListener implements KeyListener {
	
	public TrainingDataCreator parent;
	
	
	public AddPointKeyListener(TrainingDataCreator parent) {

		
		this.parent = parent;
		
	}

	@Override
	public void keyTyped(KeyEvent e) {

		if (e.getKeyChar() == 'a')

			parent.AddDot = "a";

	}

	@Override
	public void keyPressed(KeyEvent e) {

		if (e.getKeyChar() == 'a')

			parent.AddDot = "a";

	}

	@Override
	public void keyReleased(KeyEvent e) {

		if (e.getKeyChar() == 'a')

			parent.AddDot = "a";

	}

}
