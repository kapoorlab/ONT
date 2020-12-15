package pointSelector;

import java.awt.Color;

import ij.gui.OvalRoi;
import net.imglib2.RealLocalizable;

public class Roiobject {

	
	public Color color;
	
	public  OvalRoi roi;
	
	public  RealLocalizable point;
	
	public int time;
	
	
	public Roiobject(final Color color, final OvalRoi roi, final RealLocalizable point, final int time) {
		
		this.color = color;
		
		this.roi = roi;
		
		this.point = point;
		
		this.time = time;
		
	}
	
	
	public void setColor(Color color) {
		
		this.color = color;
		
	}
	
}