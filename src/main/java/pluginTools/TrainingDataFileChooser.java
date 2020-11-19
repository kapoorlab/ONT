package pluginTools;

import java.awt.CardLayout;
import java.awt.Checkbox;
import java.awt.CheckboxGroup;
import java.awt.Frame;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.border.Border;
import javax.swing.border.CompoundBorder;
import javax.swing.border.EmptyBorder;
import javax.swing.border.TitledBorder;
import fileListeners.ChooseTrainingImage;
import fileListeners.ONTManualModeListener;
import fileListeners.ONTMatlabModeListener;
import ij.ImagePlus;
import ij.WindowManager;
import io.scif.img.ImgIOException;
import loadfile.CovistoOneChFileLoader;
import pointSelector.Roiobject;


public class TrainingDataFileChooser extends JPanel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public JFrame Cardframe = new JFrame("ONEAT-TrainingDataMaker");
	public JPanel panelCont = new JPanel();
	public ImagePlus impOrig;
	public File impOrigfile;
	public JPanel panelFirst = new JPanel();
	public JPanel Panelfile = new JPanel();
	public JPanel Panelcsv = new JPanel();
	public JPanel Paneldone = new JPanel();
	public JPanel Panelrun = new JPanel();
	public final Insets insets = new Insets(10, 10, 0, 10);
	public final GridBagLayout layout = new GridBagLayout();
	public final GridBagConstraints c = new GridBagConstraints();
	public final String[] imageNames, blankimageNames;
	public JComboBox<String> ChooseImage;
	public JButton Done = new JButton("Finished choosing files, start Clicker");
	public HashMap<Integer, ArrayList<Roiobject>> MatlabOvalRois= new HashMap<Integer, ArrayList<Roiobject>>();

	public String chooseTrainDatastring = "Image for clicking";
	public Border chooseTrainData = new CompoundBorder(new TitledBorder(chooseTrainDatastring), new EmptyBorder(c.insets));

	public String chooseMatlabTrainDatastring = "Image and Matlab CSV file for correction";
	public Border chooseMatlabTrainData = new CompoundBorder(new TitledBorder(chooseTrainDatastring), new EmptyBorder(c.insets));

	public String donestring = "Done Selection";
	public Border LoadONT = new CompoundBorder(new TitledBorder(donestring), new EmptyBorder(c.insets));


	public CheckboxGroup ONTmode = new CheckboxGroup();
	public boolean ManualDots = true;
	public boolean MatlabDots = false;
	public Checkbox ManualMode = new Checkbox("Make Manual Dots", ManualDots, ONTmode);
    public Checkbox MatlabMode = new Checkbox("Select Matlab Dots", MatlabDots, ONTmode);


	public TrainingDataFileChooser() {

		panelFirst.setLayout(layout);

		Paneldone.setLayout(layout);
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

		
		panelFirst.add(Panelcsv, new GridBagConstraints(0, 9, 3, 1, 0.0, 0.0, GridBagConstraints.WEST,
				GridBagConstraints.HORIZONTAL, insets, 0, 0));
		
		Paneldone.add(Done, new GridBagConstraints(0, 0, 3, 1, 0.0, 0.0, GridBagConstraints.WEST,
				GridBagConstraints.HORIZONTAL, new Insets(10, 10, 0, 10), 0, 0));
		Paneldone.setBorder(LoadONT);
		panelFirst.add(Paneldone, new GridBagConstraints(0, 10, 3, 1, 0.0, 0.0, GridBagConstraints.WEST,
				GridBagConstraints.HORIZONTAL, insets, 0, 0));



		// Listeneres

		ManualMode.addItemListener(new ONTManualModeListener(this));
		MatlabMode.addItemListener(new ONTMatlabModeListener(this));
		Done.addActionListener(new TrainDoneListener());
		panelFirst.setVisible(true);
		cl.show(panelCont, "1");
		Cardframe.add(panelCont, "Center");

		Cardframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		Cardframe.pack();
		Cardframe.setVisible(true);
	}

	
	

	public class TrainDoneListener implements ActionListener {

		@Override
		public void actionPerformed(ActionEvent e) {

			try {
				DoneCurrBud(Cardframe);
			} catch (ImgIOException e1) {

			}
		}

	}

	
	public void DoneCurrBud(Frame parent) throws ImgIOException {

		close(parent);

	}

	protected final void close(final Frame parent) {
		if (parent != null)
			parent.dispose();

	}
	
	public static void main(String[] args) {
		
		JFrame frame = new JFrame("");

		TrainingDataFileChooser panel = new TrainingDataFileChooser();

		frame.getContentPane().add(panel, "Center");
		frame.setSize(panel.getPreferredSize());
		
		
	}
	

}