package pluginTools;

import java.awt.CardLayout;
import java.awt.Checkbox;
import java.awt.CheckboxGroup;
import java.awt.Color;
import java.awt.Frame;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.Label;
import java.awt.Rectangle;
import java.awt.Scrollbar;
import java.awt.TextComponent;
import java.awt.TextField;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.TextEvent;
import java.awt.event.TextListener;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollBar;
import javax.swing.SwingUtilities;
import javax.swing.border.Border;
import javax.swing.border.CompoundBorder;
import javax.swing.border.EmptyBorder;
import javax.swing.border.TitledBorder;
import fileListeners.ChooseTrainingImage;
import fileListeners.MouseClickTimeListener;
import fileListeners.ONTManualModeListener;
import fileListeners.ONTMatlabModeListener;
import ij.ImageListener;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.ImageCanvas;
import ij.gui.OvalRoi;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.io.Opener;
import io.scif.img.ImgIOException;
import loadfile.CovistoOneChFileLoader;
import mpicbg.imglib.util.Util;
import net.imagej.ImageJ;
import net.imglib2.Cursor;
import net.imglib2.KDTree;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealPoint;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import pointSelector.Roiobject;

public class TrainingDataCreator extends JPanel {

	/**
	 * 
	 */
	public int thirdDimensionslider = 1;
	public int thirdDimensionsliderInit = 1;
	public int thirdDimension;
	public int thirdDimensionSize;
	public Overlay overlay;

	public static enum ValueChange {

		THIRDDIMmouse, All;

	}

	public void setTime(final int value) {
		thirdDimensionslider = value;
		thirdDimensionsliderInit = 1;
		thirdDimension = 1;

	}

	public int getTimeMax() {

		return thirdDimensionSize;
	}

	public void Clickrecorder() {

		impOrig = Reshape(impOrig);
		if (this.inputimage != null) {

			setTime(thirdDimension);

			System.out.println(MatlabOvalRois.size());
			System.out.println(MatlabOvalRois);

			ONTMouseListener mvl = new ONTMouseListener();
			mvl.run("");
			impOrig.getCanvas().addMouseListener(mvl);
			mvl.imageUpdated(impOrig);

			KeyListener kvl = new AddPointKeyListener();
			impOrig.getCanvas().addKeyListener(kvl);

			int ndims = this.inputimage.numDimensions();
			if (ndims == 3) {

				thirdDimension = 1;

				thirdDimensionSize = (int) inputimage.dimension(2);

			}

		}
	}

	public void updatePreview(final ValueChange change) {

		if (change == ValueChange.THIRDDIMmouse) {

			if (overlay == null) {

				overlay = new Overlay();
				impOrig.setOverlay(overlay);

			}

			if (MatlabOvalRois.get(thirdDimension) != null) {

				for (Roiobject roi : MatlabOvalRois.get(thirdDimension)) {

					overlay.add(roi.roi);
					roi.roi.setStrokeColor(roi.color);

				}

			}
			

		}

	}

	private static final long serialVersionUID = 1L;
	public JFrame Cardframe = new JFrame("ONEAT-TrainingDataMaker");
	public JPanel panelCont = new JPanel();
	public ImagePlus impOrig;
	public File impOrigfile;
	public JPanel panelFirst = new JPanel();
	public JPanel Panelfile = new JPanel();
	public JPanel Panelclicker = new JPanel();
	public JPanel Panelrun = new JPanel();
	public final Insets insets = new Insets(10, 10, 0, 10);
	public final GridBagLayout layout = new GridBagLayout();
	public final GridBagConstraints c = new GridBagConstraints();
	public final String[] imageNames, blankimageNames;
	public JComboBox<String> ChooseImage;
	public HashMap<Integer, ArrayList<Roiobject>> MatlabOvalRois = new HashMap<Integer, ArrayList<Roiobject>>();
	public int[] Clickedpoints = new int[2];
	public String addToName = "";
	public String chooseTrainDatastring = "Image for clicking";
	public Border chooseTrainData = new CompoundBorder(new TitledBorder(chooseTrainDatastring),
			new EmptyBorder(c.insets));

	public String chooseMatlabTrainDatastring = "Image and Matlab CSV file for correction";
	public Border chooseMatlabTrainData = new CompoundBorder(new TitledBorder(chooseTrainDatastring),
			new EmptyBorder(c.insets));

	public String clickstring = "Clicker Menu";
	public Border LoadONT = new CompoundBorder(new TitledBorder(clickstring), new EmptyBorder(c.insets));
	public Label eventname;
	public TextField eventfieldname;

	public CheckboxGroup ONTmode = new CheckboxGroup();
	public boolean ManualDots = true;
	public boolean MatlabDots = false;
	public Checkbox ManualMode = new Checkbox("Make Manual Dots", ManualDots, ONTmode);
	public Checkbox MatlabMode = new Checkbox("Select Matlab Dots", MatlabDots, ONTmode);
	public File imageDirectory = new File("");
	public String imageFilename = "";
	public RandomAccessibleInterval<FloatType> inputimage;
	public final int scrollbarSize = 1000;
	public String AddDot = "b";
	public Color AcceptColor = Color.GREEN;
	public Color RejectColor = Color.RED;
	public Label timeText = new Label("Current T = " + 1, Label.CENTER);
	public String timestring = "Current T";

	public TrainingDataCreator() {

		panelFirst.setLayout(layout);
		eventname = new Label("Event/CellType Name");
		eventfieldname = new TextField(10);
		eventfieldname.setText("Normal");

		Panelclicker.setLayout(layout);
		CardLayout cl = new CardLayout();

		panelCont.setLayout(cl);
		panelCont.add(panelFirst, "1");
		imageNames = WindowManager.getImageTitles();
		blankimageNames = new String[imageNames.length + 1];
		blankimageNames[0] = " ";

		for (int i = 0; i < imageNames.length; ++i)
			blankimageNames[i + 1] = imageNames[i];

		ChooseImage = new JComboBox<String>(blankimageNames);

		CovistoOneChFileLoader original = new CovistoOneChFileLoader(chooseTrainDatastring, blankimageNames);

		Panelfile = original.SingleChannelOption();

		panelFirst.add(Panelfile, new GridBagConstraints(0, 2, 3, 1, 0.0, 0.0, GridBagConstraints.WEST,
				GridBagConstraints.HORIZONTAL, insets, 0, 0));

		original.ChooseImage.addActionListener(new ChooseTrainingImage(this, original.ChooseImage));

		panelFirst.add(ManualMode, new GridBagConstraints(0, 3, 3, 1, 0.0, 0.0, GridBagConstraints.WEST,
				GridBagConstraints.HORIZONTAL, insets, 0, 0));
		panelFirst.add(MatlabMode, new GridBagConstraints(0, 4, 3, 1, 0.0, 0.0, GridBagConstraints.WEST,
				GridBagConstraints.HORIZONTAL, insets, 0, 0));

		panelFirst.add(Panelfile, new GridBagConstraints(0, 7, 3, 1, 0.0, 0.0, GridBagConstraints.WEST,
				GridBagConstraints.HORIZONTAL, insets, 0, 0));

		Panelclicker.add(eventname, new GridBagConstraints(0, 0, 3, 1, 0.0, 0.0, GridBagConstraints.WEST,
				GridBagConstraints.HORIZONTAL, new Insets(10, 10, 0, 10), 0, 0));
		Panelclicker.add(eventfieldname, new GridBagConstraints(0, 1, 3, 1, 0.0, 0.0, GridBagConstraints.WEST,
				GridBagConstraints.HORIZONTAL, new Insets(10, 10, 0, 10), 0, 0));
		Panelclicker.setBorder(LoadONT);
		panelFirst.add(Panelclicker, new GridBagConstraints(0, 10, 3, 1, 0.0, 0.0, GridBagConstraints.WEST,
				GridBagConstraints.HORIZONTAL, insets, 0, 0));

		// Listeneres

		ManualMode.addItemListener(new ONTManualModeListener(this));
		MatlabMode.addItemListener(new ONTMatlabModeListener(this));
		eventfieldname.addTextListener(new eventnameListener());
		panelFirst.setVisible(true);
		cl.show(panelCont, "1");
		Cardframe.add(panelCont, "Center");
		Cardframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		Cardframe.pack();
		Cardframe.setVisible(true);
	}

	public class ONTMouseListener implements MouseListener, ImageListener {

		public ONTMouseListener() {

		}

		final ImageCanvas canvas = impOrig.getWindow().getCanvas();

		@Override
		public void mouseReleased(MouseEvent e) {

		}

		@Override
		public void mousePressed(MouseEvent e) {

			getTime(impOrig);

			// Make a dot red or green
			System.out.println(AddDot);
			if (SwingUtilities.isLeftMouseButton(e) && AddDot != "a") {

				int X = canvas.offScreenX(e.getX());
				int Y = canvas.offScreenY(e.getY());
				Clickedpoints[0] = X;
				Clickedpoints[1] = Y;

				if (MatlabOvalRois.get(thirdDimension) != null) {

					ArrayList<Roiobject> ClickedPointList = MatlabOvalRois.get(thirdDimension);
					double[] location = { X, Y };

					Roiobject nearestRoi = getNearestRois(ClickedPointList, location);
					ClickedPointList.remove(nearestRoi);
					System.out.println("Original color" + nearestRoi.color);
					if (nearestRoi.color == AcceptColor)
						nearestRoi.color = RejectColor;
					if (nearestRoi.color == RejectColor)
						nearestRoi.color = AcceptColor;
					System.out.println("New color" + nearestRoi.color);
					ClickedPointList.add(nearestRoi);

					
					
					if (MatlabOvalRois.containsKey(thirdDimension)) {
						ArrayList<Roiobject> currentroi = MatlabOvalRois.get(thirdDimension);
						for (Roiobject roi : currentroi) {

							roi.roi.setStrokeColor(roi.color);

							if (overlay!= null)
								overlay.add(roi.roi);

						}
						impOrig.updateAndDraw();
					}

				}

			}

			if (SwingUtilities.isLeftMouseButton(e) && AddDot == "a") {

				AddDot = "b";
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

			thirdDimension = time;
			
			if (overlay == null) {

				overlay = new Overlay();
				impOrig.setOverlay(overlay);

			}
			else
			 overlay.clear();
			
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

			updatePreview(ValueChange.THIRDDIMmouse);
			
			ImagePlus.removeImageListener(this);
			impOrig.updateAndDraw();

			impOrig.setTitle("Active Image" + " " + "Do not close this ");
			ImagePlus.addImageListener(this);
		}
	}

	public class AddPointKeyListener implements KeyListener {

		public AddPointKeyListener() {

		}

		@Override
		public void keyTyped(KeyEvent e) {

			if (e.getKeyChar() == 'a')

				AddDot = "a";

		}

		@Override
		public void keyPressed(KeyEvent e) {

			if (e.getKeyChar() == 'a')

				AddDot = "a";

		}

		@Override
		public void keyReleased(KeyEvent e) {

			if (e.getKeyChar() == 'a')

				AddDot = "a";

		}

	}

	public static ImagePlus Reshape(ImagePlus image) {

		int channels, frames;

		ImagePlus imp = image;
		if (imp.getNChannels() > imp.getNFrames()) {
			channels = imp.getNFrames();
			frames = imp.getNChannels();

		}

		else {

			channels = imp.getNChannels();
			frames = imp.getNFrames();

		}

		imp.setDimensions(channels, imp.getNSlices(), frames);
		imp.show();

		return imp;

	}

	public static int computeScrollbarPositionFromValue(final float sigma, final float min, final float max,
			final int scrollbarSize) {
		return Util.round(((sigma - min) / (max - min)) * scrollbarSize);
	}

	public static float computeValueFromScrollbarPosition(final int scrollbarPosition, final float min, final float max,
			final int scrollbarSize) {
		return min + (scrollbarPosition / (float) scrollbarSize) * (max - min);
	}

	public static Roiobject getNearestRois(ArrayList<Roiobject> roi, double[] Clickedpoint) {

		ArrayList<Roiobject> Allrois = roi;

		Roiobject KDtreeroi = null;

		final List<RealPoint> targetCoords = new ArrayList<RealPoint>(Allrois.size());
		final List<FlagNode<Roiobject>> targetNodes = new ArrayList<FlagNode<Roiobject>>(Allrois.size());
		for (int index = 0; index < Allrois.size(); ++index) {

			Roi r = Allrois.get(index).roi;
			Rectangle rect = r.getBounds();

			targetCoords.add(new RealPoint(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0));

			targetNodes.add(new FlagNode<Roiobject>(Allrois.get(index)));

		}

		if (targetNodes.size() > 0 && targetCoords.size() > 0) {

			final KDTree<FlagNode<Roiobject>> Tree = new KDTree<FlagNode<Roiobject>>(targetNodes, targetCoords);

			final NNFlagsearchKDtree<Roiobject> Search = new NNFlagsearchKDtree<Roiobject>(Tree);

			final double[] source = Clickedpoint;
			final RealPoint sourceCoords = new RealPoint(source);
			Search.search(sourceCoords);

			final FlagNode<Roiobject> targetNode = Search.getSampler().get();

			KDtreeroi = targetNode.getValue();

		}

		return KDtreeroi;

	}

	public static int[] getNearestPoint(ArrayList<int[]> roi, double[] Clickedpoint) {

		ArrayList<int[]> Allrois = roi;

		int[] KDtreeroi = null;

		final List<RealPoint> targetCoords = new ArrayList<RealPoint>(Allrois.size());
		final List<FlagNode<int[]>> targetNodes = new ArrayList<FlagNode<int[]>>(Allrois.size());
		for (int index = 0; index < Allrois.size(); ++index) {

			int[] r = Allrois.get(index);

			targetCoords.add(new RealPoint(r[1], r[2]));

			targetNodes.add(new FlagNode<int[]>(Allrois.get(index)));

		}

		if (targetNodes.size() > 0 && targetCoords.size() > 0) {

			final KDTree<FlagNode<int[]>> Tree = new KDTree<FlagNode<int[]>>(targetNodes, targetCoords);

			final NNFlagsearchKDtree<int[]> Search = new NNFlagsearchKDtree<int[]>(Tree);

			final double[] source = Clickedpoint;
			final RealPoint sourceCoords = new RealPoint(source);
			Search.search(sourceCoords);

			final FlagNode<int[]> targetNode = Search.getSampler().get();

			KDtreeroi = targetNode.getValue();

		}

		return KDtreeroi;

	}

	public class eventnameListener implements TextListener {

		@Override
		public void textValueChanged(TextEvent e) {

			final TextComponent tc = (TextComponent) e.getSource();

			String s = tc.getText();

			if (s.length() > 0)
				addToName = s;
		}

	}

	public static void main(String[] args) {

		new ImageJ();

		File csvfile = new File("/home/kapoorlab/OLDONEAT/Movie2Normal.csv");

		ImagePlus impA = new Opener().openImage("/home/kapoorlab/OLDONEAT/NEATcsvfiles/EventMovie.tif");
		impA.show();

		JFrame frame = new JFrame("");

		TrainingDataCreator panel = new TrainingDataCreator();
		frame.getContentPane().add(panel, "Center");
		frame.setSize(panel.getPreferredSize());

	}

}