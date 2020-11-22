package fileListeners;

import java.awt.Color;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.ArrayList;

import javax.swing.SwingUtilities;

import ij.ImageListener;
import ij.ImagePlus;
import ij.gui.ImageCanvas;
import ij.gui.Overlay;
import pluginTools.TrainingDataCreator;
import pluginTools.TrainingDataCreator.ValueChange;
import pointSelector.Roiobject;
public class ONTMouseListener implements MouseListener, ImageListener {

	public TrainingDataCreator parent;
	
	public ONTMouseListener(TrainingDataCreator parent) {

		this.parent = parent;
	}

	final ImageCanvas canvas = parent.impOrig.getWindow().getCanvas();

	@Override
	public void mouseReleased(MouseEvent e) {

	}

	@Override
	public void mousePressed(MouseEvent e) {

		getTime(parent.impOrig);

		// Make a dot red or green
		System.out.println(parent.AddDot);
		if (SwingUtilities.isLeftMouseButton(e) && parent.AddDot != "a") {

			int X = canvas.offScreenX(e.getX());
			int Y = canvas.offScreenY(e.getY());
			parent.Clickedpoints[0] = X;
			parent.Clickedpoints[1] = Y;

			if (parent.MatlabOvalRois.get(parent.thirdDimension) != null) {

				ArrayList<Roiobject> ClickedPointList = parent.MatlabOvalRois.get(parent.thirdDimension);
				double[] location = { X, Y };

				Roiobject nearestRoi = TrainingDataCreator.getNearestRois(ClickedPointList, location);
				ClickedPointList.remove(nearestRoi);
				System.out.println("Original color" + nearestRoi.color);
				
				Color newcolor;
				if (nearestRoi.color == parent.AcceptColor)

					newcolor = parent.RejectColor;
				else
					
					newcolor = parent.AcceptColor;
				nearestRoi.color = newcolor;
				System.out.println("New color" + nearestRoi.color);
				ClickedPointList.add(nearestRoi);

				parent.MatlabOvalRois.put(parent.thirdDimension, ClickedPointList);
				
				if (parent.MatlabOvalRois.containsKey(parent.thirdDimension)) {
					ArrayList<Roiobject> currentroi = parent.MatlabOvalRois.get(parent.thirdDimension);
					for (Roiobject roi : currentroi) {

						roi.roi.setStrokeColor(roi.color);

						if (parent.overlay!= null)
							parent.overlay.add(roi.roi);

					}
					parent.impOrig.updateAndDraw();
				}

			}

		}

		if (SwingUtilities.isLeftMouseButton(e) && parent.AddDot == "a") {

			parent.AddDot = "b";
		}
	}

	@Override
	public void mouseExited(MouseEvent arg0) {
	}

	@Override
	public void mouseEntered(MouseEvent arg0) {
	}

	@Override
	public void mouseClicked(MouseEvent arg0) {
	}

	public void getTime(ImagePlus imp) {
		
		
		int time = imp.getFrame();

		parent.thirdDimension = time;
		
		if (parent.overlay == null) {

			parent.overlay = new Overlay();
			parent.impOrig.setOverlay(parent.overlay);

		}
		else
		 parent.overlay.clear();
		
	}

	public void run(String arg) {
		ImagePlus.addImageListener(this);
	}

	// called when an image is opened
	public void imageOpened(ImagePlus imp) {
	}

	// Called when an image is closed
	public void imageClosed(ImagePlus imp) {
	}

	// Called when an image's pixel data is updated
	public void imageUpdated(ImagePlus imp) {

		getTime(imp);

		parent.updatePreview(ValueChange.THIRDDIMmouse);
		
		ImagePlus.removeImageListener(this);
		parent.impOrig.updateAndDraw();

		parent.impOrig.setTitle("Active Image" + " " + "Do not close this ");
		ImagePlus.addImageListener(this);
	}
}
