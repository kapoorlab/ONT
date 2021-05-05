package fileListeners;

import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

import pluginTools.TrainingDataCreator;
import pluginTools.TrainingDataCreator.ValueChange;


public class MouseClickStandardMouseListener implements MouseListener
{
	final TrainingDataCreator parent;
	final ValueChange change;

	public MouseClickStandardMouseListener( final TrainingDataCreator parent, final ValueChange change)
	{
		this.parent = parent;
		this.change = change;
	}
	
	

	@Override
	public void mouseReleased( MouseEvent arg0 )
	{
		
		
		parent.updatePreview(change);
		

		
	}

	@Override
	public void mousePressed( MouseEvent arg0 ){
		
		
	
	}

	@Override
	public void mouseExited( MouseEvent arg0 ) {
	
	}

	@Override
	public void mouseEntered( MouseEvent arg0 ) {
	}

	@Override
	public void mouseClicked( MouseEvent arg0 ) {}
}



