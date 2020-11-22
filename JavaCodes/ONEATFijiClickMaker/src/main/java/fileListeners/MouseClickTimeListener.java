package fileListeners;

import java.awt.Color;
import java.awt.Label;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import java.util.ArrayList;

import javax.swing.JScrollBar;

import ONEATkalmanGUI.CovistoKalmanPanel;
import ij.IJ;
import ij.gui.OvalRoi;
import mpicbg.imglib.util.Util;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.logic.BitType;
import pluginTools.TrainingDataCreator;
import pluginTools.TrainingDataCreator.ValueChange;
import pointSelector.Roiobject;

public class MouseClickTimeListener implements AdjustmentListener {
	final Label label;
	final String string;
	TrainingDataCreator parent;
	final float min, max;
	final int scrollbarSize;

	final JScrollBar deltaScrollbar;

	public MouseClickTimeListener(final TrainingDataCreator parent, final Label label, final String string, final float min, final float max,
			final int scrollbarSize, final JScrollBar deltaScrollbar) {
		this.label = label;
		this.parent = parent;
		this.string = string;
		this.min = min;
		this.max = max;
		this.scrollbarSize = scrollbarSize;

		this.deltaScrollbar = deltaScrollbar;
		//deltaScrollbar.addMouseMotionListener(new BudMouseListener(parent, ValueChange.THIRDDIMmouse));
		deltaScrollbar.addMouseListener(new MouseClickStandardMouseListener(parent, ValueChange.THIRDDIMmouse));
		deltaScrollbar.setBlockIncrement(computeScrollbarPositionFromValue(2, min, max, scrollbarSize));
		deltaScrollbar.setUnitIncrement(computeScrollbarPositionFromValue(2, min, max, scrollbarSize));
	}

	@Override
	public void adjustmentValueChanged(AdjustmentEvent e) {
		
		
		parent.thirdDimension = (int) Math.round(computeValueFromScrollbarPosition(e.getValue(), min, max, scrollbarSize));

		if(parent.impOrig.getOverlay()!=null) 
		parent.impOrig.getOverlay().clear();
		deltaScrollbar
		.setValue(computeScrollbarPositionFromValue(parent.thirdDimension, min, max, scrollbarSize));
		
		label.setText(string +  " = "  + parent.thirdDimension);

		parent.panelFirst.validate();
		parent.panelFirst.repaint();
		
		

		if(parent.MatlabOvalRois.containsKey(parent.thirdDimension)) {
			ArrayList<Roiobject> currentroi = parent.MatlabOvalRois.get(parent.thirdDimension);
			for(Roiobject roi:currentroi) {
			roi.roi.setStrokeColor(roi.color);
			parent.impOrig.getOverlay().add(roi.roi);
		
			
			}
			
		}
		
			
			parent.impOrig.updateAndDraw();
		
	

	}
	public static int computeScrollbarPositionFromValue(final float sigma, final float min, final float max,
			final int scrollbarSize) {
		return Util.round(((sigma - min) / (max - min)) * scrollbarSize);
	}
	public static float computeValueFromScrollbarPosition(final int scrollbarPosition, final float min, final float max,
			final int scrollbarSize) {
		return min + (scrollbarPosition / (float) scrollbarSize) * (max - min);
	}



}