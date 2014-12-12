import numpy as np
from astropy.io import fits as pyfits
from scipy import ndimage
from scipy.ndimage.filters import maximum_filter1d as maxfilt
from scipy.io import readsav
import os
import scipy.interpolate as scin
from scipy.linalg import solve_banded
import scipy.ndimage.filters as filters
from scipy.linalg.fblas import dgemm
from astropy.modeling import models, fitting
from astropy.stats.funcs import sigma_clip
from matplotlib import pyplot as plt
import Image, ImageDraw
import sys
import math, random
from itertools import product
from warnings import warn #just for least_squares_solver

def least_squares_solver(a, b, residuals=False):
	#	Inputs
	#	a : (M, N) array_like "Coefficient" matrix.
	#	b : (M,) array_like ordinate or "dependent variable" values.
	#	Returns
	#	x : (M,) ndarray
	#   Least-squares solution. The shape of `x` depends on the shape of `b`.
	#	residuals : int (Optional)
	#   Sums of residuals: squared Euclidean 2-norm for each column in ``b - a*x``.
    if type(a) != np.ndarray or not a.flags['C_CONTIGUOUS']:
       warn('Matrix a is not a C-contiguous numpy array. The solver will create a copy, which will result' + \
            ' in increased memory usage.')
    a = np.asarray(a, order='C')
    i = dgemm(alpha=1.0, a=a.T, b=a.T, trans_b=True)
    x = np.linalg.solve(i, dgemm(alpha=1.0, a=a.T, b=b)).flatten()
    if residuals:
        return x, np.linalg.norm(np.dot(a, x) - b)
    else:
        return x

class UFarray:
    def __init__(self):
        # Array which holds label -> set equivalences
        self.P = []

        # Name of the next label, when one is created
        self.label = 0

    def makeLabel(self):
        r = self.label
        self.label += 1
        self.P.append(r)
        return r
    
    # Makes all nodes "in the path of node i" point to root
    def setRoot(self, i, root):
        while self.P[i] < i:
            j = self.P[i]
            self.P[i] = root
            i = j
        self.P[i] = root

    # Finds the root node of the tree containing node i
    def findRoot(self, i):
        while self.P[i] < i:
            i = self.P[i]
        return i
    
    # Finds the root of the tree containing node i
    # Simultaneously compresses the tree
    def find(self, i):
        root = self.findRoot(i)
        self.setRoot(i, root)
        return root
    
    # Joins the two trees containing nodes i and j
    # Modified to be less agressive about compressing paths
    # because performance was suffering some from over-compression
    def union(self, i, j):
        if i != j:
            root = self.findRoot(i)
            rootj = self.findRoot(j)
            if root > rootj: root = rootj
            self.setRoot(j, root)
            self.setRoot(i, root)
    
    def flatten(self):
        for i in range(1, len(self.P)):
            self.P[i] = self.P[self.P[i]]
    
    def flattenL(self):
        k = 1
        for i in range(1, len(self.P)):
            if self.P[i] < i:
                self.P[i] = self.P[self.P[i]]
            else:
                self.P[i] = k
                k += 1

def cclabel(img):
    print'Running connected component method'
    data=img
    width, height = data.shape
 
    # Union find data structure
    uf = UFarray()
 
    #
    # First pass
    #
 
    # Dictionary of point:label pairs
    labels = {}
 
    for y, x in product(range(height), range(width)):
 
        #
        # Pixel names were chosen as shown:
        #
        #   -------------
        #   | a | b | c |
        #   -------------
        #   | d | e |   |
        #   -------------
        #   |   |   |   |
        #   -------------
        #
        # The current pixel is e
        # a, b, c, and d are its neighbors of interest
        #
        # 255 is white, 0 is black
        # White pixels part of the background, so they are ignored
        # If a pixel lies outside the bounds of the image, it default to white
        #
 
        # If the current pixel is white, it's obviously not a component...
        if data[x, y] == 255:
            pass
 
        # If pixel b is in the image and black:
        #    a, d, and c are its neighbors, so they are all part of the same component
        #    Therefore, there is no reason to check their labels
        #    so simply assign b's label to e
        elif y > 0 and data[x, y-1] == 0:
            labels[x, y] = labels[(x, y-1)]
 
        # If pixel c is in the image and black:
        #    b is its neighbor, but a and d are not
        #    Therefore, we must check a and d's labels
        elif x+1 < width and y > 0 and data[x+1, y-1] == 0:
 
            c = labels[(x+1, y-1)]
            labels[x, y] = c
 
            # If pixel a is in the image and black:
            #    Then a and c are connected through e
            #    Therefore, we must union their sets
            if x > 0 and data[x-1, y-1] == 0:
                a = labels[(x-1, y-1)]
                uf.union(c, a)
 
            # If pixel d is in the image and black:
            #    Then d and c are connected through e
            #    Therefore we must union their sets
            elif x > 0 and data[x-1, y] == 0:
                d = labels[(x-1, y)]
                uf.union(c, d)
 
        # If pixel a is in the image and black:
        #    We already know b and c are white
        #    d is a's neighbor, so they already have the same label
        #    So simply assign a's label to e
        elif x > 0 and y > 0 and data[x-1, y-1] == 0:
            labels[x, y] = labels[(x-1, y-1)]
 
        # If pixel d is in the image and black
        #    We already know a, b, and c are white
        #    so simpy assign d's label to e
        elif x > 0 and data[x-1, y] == 0:
            labels[x, y] = labels[(x-1, y)]
 
        # All the neighboring pixels are white,
        # Therefore the current pixel is a new component
        else: 
            labels[x, y] = uf.makeLabel()
 
    #
    # Second pass
    #
 
    uf.flatten()
 
    colors = {}

    # Image to display the components in a nice, colorful way
    output_img = Image.new("RGB", (width, height))
    outdata = output_img.load()

    for (x, y) in labels:
 
        # Name of the component the current point belongs to
        component = uf.find(labels[(x, y)])

        # Update the labels with correct information
        labels[(x, y)] = component
 
        # Associate a random color with this component 
        if component not in colors: 
            colors[component] = (random.randint(0,255), random.randint(0,255),random.randint(0,255))

        # Colorize the image
        outdata[x, y] = colors[component]

    return (labels)

def trace_orders():
	figure=False
	print'Tracing orders'
	flat=pyfits.open('radius_masterflat.fits')
	flat=flat[0].data
	flat=flat.T
	smoothed=np.zeros_like(flat)
	a,b=smoothed.shape
	#primary smoothing
	for i in np.arange(1,a):
		smoothed[i,:]=ndimage.filters.gaussian_filter1d(flat[i,:],25)
	#sobel operator
	dx=ndimage.sobel(smoothed,0)
	#run connected component analysis

	filtered=np.ones_like(flat)
	zeros=np.zeros_like(flat)
	filtered[np.where(np.abs(dx)<0.058*np.max(smoothed))]=zeros[np.where(np.abs(dx)<0.058*np.max(smoothed))]
	filtered=filtered*255

	labels=cclabel(filtered)

	components={} #preset data structure
	for key, value in labels.iteritems():
		if value not in components:
			components[value]=[]
			components[value].append(key)
		else:
			components[value].append(key)

	#filter for good orders
	orderparameters=[]
	blank=np.zeros_like(flat)
	for i in components:
		xx=[]
		yy=[]
		if len(components[i])>b*5:
			for j in components[i]:
				blank[j]=1.
				#fit polynomials to orders
				c,d=j
				xx.append(c)
				yy.append(d)
			coefficients=np.polyfit(yy,xx,4) #fit
			orderparameters.append(coefficients)
	
	orderparameters[0]=[]
	orderparameters[-1]=[]
	

	np.save('order_definition_parameters',orderparameters)
	if figure==True:
		plt.subplot(2,2,1)
		plt.imshow(flat)
		plt.colorbar()
		plt.subplot(2,2,2)
		plt.imshow(dx)
		plt.colorbar()
		plt.subplot(2,2,3)
		plt.imshow(blank)
		plt.subplot(2,2,4)
		plt.imshow(flat)
		for ord in orderparameters:
			if ord!=[]:
				yyy=np.polyval(ord,np.arange(flat.shape[1]))
				plt.plot(np.arange(flat.shape[1]),yyy,color='red')
				#if ord!=[]: print ord[-1],np.min(yyy)
		plt.show()
	return orderparameters

def findscatter(file='radius_masterflat.fits',nsteps = 100,poly_order_x = 6,poly_order_y = 4,figures=False, method='single'):
	print 'Determining Scattered Light'
	figures=False
	hdulist=pyfits.open(file)
	inputimage=hdulist[0].data
	#order_definition_parameters.npy
	orderparameters=np.load('order_definition_parameters.npy')

	max_gradient = 5
	a,b=inputimage.shape
	if hdulist[0].header['FOCALMOD']!='TEKTRONIX':
		if b>a:
			inputimage=np.transpose(inputimage)
		a,b=inputimage.shape
		#print 'a>b unless focalmod == tektronix. a= %s, b= %s, %s' %(a,b,hdulist[0].header['FOCALMOD'])
	xrange=np.arange(0,a,1)
	yrange=np.arange(0,b,1)
	#trace_order_params(xrange,orderparameters)
	
	stepsize=(100/nsteps)
	xslices=[]
	for i in np.round(np.arange(0,100,stepsize)):
		#get xslice locations
		if i==0:
			xslices=np.round(np.percentile(xrange,i))
		else:
			xslices=np.hstack((xslices,np.round(np.percentile(xrange,i))))
	yslices=[]
	for i in np.round(np.arange(5,95,stepsize)):
		#get xslice locations
		if i==0:
			yslices=np.round(np.percentile(yrange,i))
		else:
			yslices=np.hstack((yslices,np.round(np.percentile(yrange,i))))
	

	#reconstruct ordtrace
	for ord, item in enumerate(orderparameters):
		if item != []:
			temp=np.polyval(orderparameters[ord],xrange)#previously [ord][0]
			try: 
				fitted_orders
			except NameError:
				fitted_orders=temp
			else:
				fitted_orders=np.vstack((fitted_orders,temp))
	fitted_orders=np.sort(fitted_orders,axis=0)
	#have now reconstructed ordtrace
	
	#fig = plt.figure(1,figsize=(12,8))
	#plt.imshow(masterflatdata.T)
	#for row in fitted_orders:
	#	yy=np.round(row)
	#	yy=yy.astype('int64')
	#	plt.scatter(xrange,row,color='red')
	#plt.show()
	
	#print fitted_orders


	#make starting array of troughs
	a,b=fitted_orders.shape
	troughs=np.zeros((a-1,b))
	for rowind, row in enumerate(fitted_orders):
		for x in xrange:
			if rowind < len(fitted_orders)-1:
				#last 'order' is in overscan
				troughs[rowind,x]=np.mean((fitted_orders[rowind+1,x],row[x]))
				slice=inputimage[x,:]
				#needs to be the index at the minimum on the image
				#adjustment=np.argmin(slice[row[x]-2:row[x]+2])
				#troughs[rowind,x]=troughs[rowind,x]+(adjustment-2)
	#print "found troughs - now fitting polynomials"
	

	#for row in troughs:
	#	yy=np.round(row)
	#	yy=yy.astype('int64')
	#	plt.plot(xrange,row,color='red')

	
	
	
	coefficients=[]
	for ind, row in enumerate(troughs):
		if row !=[] or ind==len(troughs)-1:
			itercount=0
			xx=xrange
			yy=row
			while 1:
				coeffs=np.polyfit(xx,yy,4)
				vals=np.polyval(coeffs,xx) #variances = data - model
				variances=yy-vals
				maskedvars = sigma_clip(variances,3,1) #sigma clip 3sd 
				mask=maskedvars.mask
				if itercount==2 or np.sum(mask)==0:
					break
				#print "sigma-clipping %s outlier(s) from order" %(np.sum(mask))
				xx=np.delete(xx,mask)
				yy=np.delete(yy,mask)
				itercount=itercount+1
		
			if coefficients==[]:
				coefficients=coeffs			
			else:
				if not coeffs.all==0:
					coefficients=np.vstack((coefficients,coeffs))
	coefficients=coefficients[:-1,:]		
	#have done with finding and refitting troughs
	
	#make array of trough values at slices
	
	trough_values=np.zeros((len(xslices),len(coefficients)))
	for sliceind,xslice in enumerate(xslices):
		for coeffind,coeff in enumerate(coefficients):
			trough_values[sliceind,coeffind]=np.polyval(coeff,xslice)
	
			
	if figures==True:
		plt.figure(1,figsize=(8,8),aspect=1.4)
		plt.subplot(2,2,1)

		plt.title("Inter-Order Troughs")
		plt.imshow(masterflatdata.T)
		xx=xslices#??
		for ind,x in enumerate(xx):
			xvals=[x]*len(trough_values[ind])
			plt.scatter(xvals,trough_values[ind],color='red',  linewidths=0, marker='.')






	firstmins=np.zeros((len(xslices)))
	#now to polyfit to image values per column of these trough_value indices
	#vert slice trough polyfitting at xslice locations
	inputimage=inputimage.T
	vslice_coeffs=[]*nsteps
	for ind, x in enumerate(xslices):
		yvals=np.round(trough_values[ind])
		yvals=yvals.astype('int64')
		xx=np.round(x)
		xx=xx.astype('int64')
		image_vals=np.zeros(yvals.shape)
		for yind,ycoord in enumerate(yvals):
			image_vals[yind]=np.min(inputimage[[ycoord-2,ycoord-1,ycoord,ycoord+1,ycoord+2],[xx-2,xx-1,xx,xx+1,xx+2]])
			#zvals[ind,yind]=image_vals[yind]
		count=0
		
		#plot image_vals for moire pattern before smoothing
		
		# moving minimum and smoothing operation to deal with narrowly spaced order
		# minimum cyclicity.
		windowsize = 10 #pixels for moving max and min calc
		minima = filters.minimum_filter1d(image_vals, windowsize, mode='reflect')
		kern=np.ones(windowsize*2.5)/(windowsize*2.5)
		image_vals = filters.convolve(minima,kern, mode='nearest')

		while 1:
			vcoeffs=np.polyfit(yvals,image_vals,poly_order_y)
			vals=np.polyval(vcoeffs,yvals) #variances = data - model
			variances=yvals-vals
			maskedvars = sigma_clip(variances,3,1) #sigma clip 3sd 
			mask=maskedvars.mask
			if count==1 or np.sum(mask)==0:
				break
			#print "sigma-clipping %s outlier(s) from order" %(np.sum(mask))
			yvals=np.delete(yvals,mask)
			image_vals=np.delete(image_vals,mask)
			itercount=itercount+1
	
		if not coeffs.all==0:
			vslice_coeffs.append(vcoeffs)


	#ax = fig.gca(projection='3d')
	#for ind,x in enumerate(xslices):
	#	xx=[x]*len(yrange)	
	#	z=np.polyval(vslice_coeffs[ind],yrange)
	#	ax.plot(xx,yrange,z,color='black')

	
	#now lets do a last set of fits in the other direction - nearly there
	inputimage=inputimage.T
	#will have to iterate over the whole xrange,yrange when we are done to make a fill image
	
	zvals=np.zeros((len(vslice_coeffs),len(yrange)))
	
	# what follows is the two-pass method
	for yind,y in enumerate(yrange):
		for sliceind,coeffs in enumerate(vslice_coeffs):
			zvals[sliceind,yind]=np.polyval(coeffs,y)
	
	if method=='double':

	#fig = plt.figure(1,figsize=(12,8))	
	#plt.imshow(zvals)
	#plt.show() 		
	
		image_coeffs=[]
		for y in yrange:
			itercount=0
			xinds=xslices
			scattervals=zvals[:,y]
			while 1:
				scattercoeffs=np.polyfit(xinds,scattervals,poly_order_x)
				newvals=np.polyval(scattercoeffs,xinds)
				variances=scattervals-newvals
				maskedvars = sigma_clip(variances,3,1) #sigma clip 3sd 
				mask=maskedvars.mask
				if itercount==2 or np.sum(mask)==0:
					break
				print "sigma-clipping %s outlier(s) from order" %(np.sum(mask))
				xinds=np.delete(xinds,mask)
				scattervals=np.delete(scattervals,mask)
				itercount=itercount+1
		
			if not scattercoeffs.all==0:
				#shouldn't occur really
				image_coeffs.append(scattercoeffs)	
		if figures==True:	
			fig = plt.figure(1,figsize=(12,8))
			ax = fig.gca(projection='3d')
			for ind,i in enumerate(image_coeffs):
				z=np.polyval(i,xrange)
				y=[yrange[ind]]*len(xrange)
				ax.plot(xrange,y,z,color='black')
			plt.show()
	
	#now reconstruct the full scatter image
		scatter_image=np.zeros((len(xrange),len(yrange)))
		#print'Construction of scatter image underway'
		#print'This will take a few minutes'
		for xxx in xrange:
			for yyy in yrange:
				scatter_image[xxx,yyy]=np.polyval(image_coeffs[yyy],xxx)
	
	###NOTE last now 1034 may be really bad
	
	# single-pass method
	if method=='single':
	# two dimensional polynomial surface fitting. The order of the polynomials
	# in each direction may be set as args and the output array sized too.
	# outarrays are indices of points to calculate the fit for once coefficients 
	# are found.
	
		

		x, y = np.meshgrid(xslices,yrange)
		z=zvals.T

		xx, yy = np.meshgrid(xrange,yrange)

		# Fit the data using astropy.modeling
		p_init = models.Polynomial2D(degree=6)
		f = fitting.LinearLSQFitter()
		p = f(p_init, x, y, z)
		scatter_image=p(xx,yy)
		# Plot the data with the best-fit model
		if figures==True:
			plt.subplot(2,2,2)
			plt.scatter(x, y, c=z, linewidths=0, marker='.')
			plt.title("Smoothed Sampled Data")
			plt.subplot(2,2,3)
			plt.scatter(xx,yy,c=scatter_image, linewidths=0, marker='.')
			plt.title("Model")
			plt.subplot(2,2,4)
			plt.scatter(xx,yy,c=inputimage.T - scatter_image,  linewidths=0, marker='.')
			plt.title("Residual")
			plt.show()
		
		scatter_image=p(xx,yy)
		
	if figures==True:
		fig = plt.figure(22,figsize=(8,6))
		ax = Axes3D(fig)
		X,Y=np.meshgrid(xrange,yrange)
		ax.plot_wireframe(X,Y,scatter_image,rstride=20,cstride=30)
		plt.title("A Smooth scattered light model")
	#ax.plot_wireframe(X,Y,masterflatdata.T,rstride=20,cstride=30)	
		plt.show()
	

	#np.save('scattered_light_model', scatter_image)	
	return scatter_image 
	
def flat_sp_func(order,ycen,osample,lamb_sp,lambda_sf,use_mask=0,noise=5.85,uncert=False,im_output=False,normflat=False,slitfunc=False):

	#Tu mamy wczytany gotowy rzad w tablice "order"
	#Obtain dimensions of the order array
	nrow,ncol=order.shape
    #noise=0.0 #Noise should be asked as an input parameter above with default=0		  
	n=(nrow+1)*osample+1
	#test=np.zeros((ncol,2))
	#If no mask
	try:
		if (use_mask==0):
			#print'generating mask'
    		#tworzymy maske, jako ze nie jest zdefiniowana
    		#create a mask, as it is not defined
			mask=np.ones((nrow, ncol))
			#j= np.where(order > 39999) #assumes divided by flat??
			mask[np.where(order > 60000)]=0.0
	except:
		#print 'using supplied mask'
		mask=np.copy(use_mask)
	#Sprawdzam, jak jest zdefiniowane lamb_sf
	#Checking for a defined lamb_sf
	if (lambda_sf==0.): 
		lambda_sf=0.1
	y=np.arange(n)/float(osample)-1.
	bklind=np.arange(osample+1)+n*osample
	oind=np.arange(osample+1)*(osample+2)
	olind=oind[0:osample+1]
	for m in range(osample+1, 2*osample+1):
		mm=m-osample
		bklind=np.append(bklind, np.arange(osample+1-mm)+n*m)
		olind=np.append(olind, oind[0:osample-mm+1]+mm)
	#Teraz idzie zgadywane widmo - na modle REDUCE
	#jako juz finalna postac jednowymiarowa
	#Now create spectrum - with REDUCE
	sp=np.repeat(0,ncol)
	sf=np.sum(order*mask,axis=1)
	#Konstrukcja pierwszego oszcowaniea slit function i widma
	#Liczenie mediany z piecioelementowych slotow - wedle procedur w reduce
	#Construction of the first slit function and spectrum
	#Counting median widths - according to the REDUCE procedure
	#Szacujemy pierwsza postac slit function
	#We estimate the first part of the slit function
	sf_med=np.arange(sf.shape[0]-4)
	for i in range(2, sf.shape[0]-2):
		sf_med[i-2]=np.median(sf[i-2:i+3])
	sf[2:sf.shape[0]-2]=sf_med
	sf=sf/np.sum(sf)
	#Szacujemy pierwsza postac widma
	#We estimate the first character of the spectrum
	sp=np.sum((order*mask)*(np.outer(sf,np.repeat(1,ncol))),axis=0)
	sp_med=np.arange(sp.shape[0]-4)
	for i in range(2, sp.shape[0]-2):# use scipy smoothing kernel later
		sp_med[i-2]=np.median(sp[i-2:i+3])
	sp[2:sp.shape[0]-2]=sp_med
	sp=sp/np.sum(sp)*np.sum(order*mask)#these sums were calculated prior...
	dev = np.sqrt(np.sum(mask*(order-np.outer(sf, sp))**2)/np.sum(mask))
	#j= np.where(abs(order-np.outer(sf, sp)) > 3.*dev)
	mask[np.where(abs(order-np.outer(sf, sp)) > 3.*dev)]=0.0
	#Wyznaczemie wagi
	#Calculate weights
	weight=1./np.float64(osample)

########################################################################
	for iter in range(1,25):
	#if iter == 2: gold = np.copy(sp)
	#teraz budujemy macierz, gdzie omega bedzie na przekatnej
	#build a matrix, where Omega will be the diagonal
		Akl=np.zeros((2*osample+1,n))
		Bl=np.zeros((1,n))
		omega=np.repeat(weight,osample+1)
		for i in range(0, ncol):
			#Tworzenie tablic wag, nie wymaga czytania rzedu
			#Creating arrays by weight, does not require reading trace
			omega=np.repeat(weight,osample+1)
			yy=y+ycen[i]
			ind=np.where((yy>=0.0) & (yy<1.))[0]
			i1=ind[0]
			i2=ind[-1]
			omega[0]=yy[i1]#Zmieniamy pierwsza wage w obrebie piksela
						   #Change the first wage within the pixel
			omega[-1]=1.-yy[i2]#Zmieniamy ostatnia wage w obrebie piksela
							   #Change the last wage within the pixel        
			#Ostateczna postac macierzy omega dla pojedynczego piksela
			#The final figure of omega matrix for each pixel
			o=np.outer(omega,omega) #najpierw transponujemy, bo w idlu jest inna notacja wymiarow macierzy (kolumna, wers)
									#first transpose, because idlu is another notation dimensional matrix (column, line)
			o[osample,osample]=o[osample,osample]+o[0,0] #zrobione wedle tego, co jest w reduce
														 #done according to what is in the reduce        
			bkl=np.zeros((2*osample+1,n))
			omega_t=np.reshape(o, o.shape[0]*o.shape[1])
			oo= omega_t[olind]
			for l in range(0, nrow):
				bkl_temp=np.reshape(bkl, bkl.shape[0]*bkl.shape[1])
				t=l*osample+bklind+i1
				bkl_temp[t]=oo*mask[l,i]
			bkl=np.reshape(bkl_temp,(2*osample+1,n))
			oo=o[osample, osample]
			for l in range(1, nrow):
				bkl[osample,l*osample+i1]=oo*mask[l,i]
			bkl[osample,nrow*osample+i1]=omega[osample]**2*mask[nrow-1,i]
			for m in range (0,osample):
				bkl[m,(osample-m):(n)]=bkl[2*osample-m,0:(n-osample+m)]
			Akl=Akl+(sp[i]**2)*bkl
		#Tu teoretycznie mamy Akl jak z artykulu o reduce rownianie 10
		#powinno byc poprawnie powyzej
		#Here we theoretically Akl article about how to reduce a par 10
		#should be properly above	
		#Teraz robimy macierz Bkl, ktora w artykule jest zapisana jako macierz R
		#Now we Bkl matrix, which the article has written as matrix R_k
			o=np.zeros((1,n))
			for l in range (0, nrow):
				o[0,l*osample+i1:l*osample+i1+osample+1]=order[l,i]*weight*mask[l,i]
			for l in range (1, nrow):
				o[0,l*osample+i1]=order[l-1,i]*omega[osample]*mask[l-1,i]+order[l,i]*omega[0]*mask[l,i]
			o[0,i1]=order[0,i]*omega[0]*mask[0,i]
			o[0,nrow*osample+i1]=order[nrow-1,i]*omega[osample]*mask[nrow-1,i]
			Bl=Bl+sp[i]*o
		#Koniec petli po i
		#mamy prawa strone rowania 8, zdefiniowana rownaniem 1
		#The end of the loop, and we have eight right side of the equation, defined equation, one
		tab=np.zeros((n,2))
		#definiujemy czynnik lambda
		lamda=lambda_sf*np.sum(Akl[osample,:])/n
		lambda_tab=np.zeros((1,n))
		for elem in range(0,n):
			lambda_tab[0,elem]=lamda
			#wkladamy czynnik lamda do tablicy Akl
			#rownowazne tablicy Bjk w pracy o reduce
			# put our factor lamda Akl equivalent to an array of array Bjk reduce the work of		
		Akl[osample,0]=Akl[osample,0]+lambda_tab[0,0]
		Akl[osample,n-1]=Akl[osample,n-1]+lambda_tab[0,n-1]
		Akl[osample,1:n-1]=Akl[osample,1:n-1]+2.*lambda_tab[0,1:n-1]
		Akl[osample+1,0:n-1]=Akl[osample+1,0:n-1]-lambda_tab[0,0:n-1]
		Akl[osample-1,1:n]=Akl[osample-1,1:n]-lambda_tab[0,1:n]
		Bl=Bl.T
		x = solve_banded((osample,osample), Akl, Bl, overwrite_ab=True, overwrite_b=True)
		ind0=[np.where(x<0)]
		x[ind0]=0.0
		sf=x/np.sum(x)*osample
		r=np.repeat(0.,sp.shape[0])
		sp_old=np.copy(sp)
		dev_new=0.0
		for i in range(0, ncol):
			omega=np.repeat(weight,osample)
			yy=y+ycen[i]
			ind1=np.where((yy>=0.0) & (yy<nrow))[0]
			i1=ind1[0]
			i2=ind1[-1]
			omega[0]=yy[i1]
			ssf=np.reshape(sf[i1:i2+1],(nrow, osample))
			o=np.dot(ssf,omega)
			#figures=True #plot slit funcs in iteration
			#if figures==True:
			#	if i>5:
			#		plt.plot(ssf)
			#		plt.show()
			yyy=nrow-yy[i2]
			o[0:nrow-1]=o[0:nrow-1]+ssf[1:nrow,0]*yyy
			o[nrow-1]=o[nrow-1]+sf[i2]*yyy
			r[i]=np.dot((order[:,i]*mask[:,i]),o)
			sp[i] = np.sum( o**2 * mask[:,i])
			if (iter > 1):
				norm=r[i]/sp[i]
				j= np.where(abs(order[:,i]-np.transpose(norm*o))>7.*dev)		
				mask[j,i]=0.0			
		
				dev_new=dev_new+np.sum(mask[:,i]*(order[:,i]-np.transpose(norm*o))**2)
		if (iter >1 ):
			dev=np.sqrt(noise**2+dev_new/np.sum(mask))
		if (lamb_sp != 0.0):
			lamda=lamb_sp*np.sum(sp)/ncol
			ab=np.zeros((3,ncol))
			ab[0,1:]=-lamda
			ab[2,:-1]=-lamda
			ab[1,0]=lamda+1.
			ab[1,-1]=lamda+1.
			ab[1,1:-1]=2.*lamda+1.
			sp=solve_banded((1,1), ab, r/sp, overwrite_ab=False, overwrite_b=False)
		else:
			sp = r/sp
		if ((abs(sp-sp_old)/sp.max()).max()<0.00001):
			break
	
	#have a look at im_out once done i guess - reconstructed image?
	jbad=np.array(0,dtype=np.int64)
	unc=np.repeat(0.,ncol)
	im_out=np.zeros_like((order))
	slitfunc_out=np.zeros_like((order))
	for i in range(0, ncol):
		omega=np.repeat(weight,osample)
		yy=y+ycen[i]
		ind1=np.where((yy>=0.0) & (yy<nrow))[0]
		i1=ind1[0]
		i2=ind1[-1]
		omega[0]=yy[i1]
		ssf=np.reshape(sf[i1:i2+1],(nrow, osample))
		o=np.dot(ssf,omega)		
		yyy=nrow-yy[i2]
		o[0:nrow-1]=o[0:nrow-1]+ssf[1:nrow,0]*yyy
		o[nrow-1]=o[nrow-1]+sf[i2]*yyy
		j = np.where((abs(order[:,i]-np.transpose(sp[i]*o))).flatten()<5*dev)
		b = np.where((abs(order[:,i]-np.transpose(sp[i]*o))).flatten()>=5*dev)
		nj=sp[j].shape[0] #length vector of good spatial values
		#done in three steps iteratively b are outliers I think
		if (nj< nrow):
			jbad=np.append(jbad, nrow*i+b[0])
		if (nj>2):
			ss=np.sum((order[j,i]-sp[i]*o[j])**2)
			xx=np.sum((o[j]-np.mean(o[j]))**2)*(nj-2)
			unc[i]=ss/xx
		else:
			unc[i]=0.0
		#is this error?? unc == uncertainty?
		im_out[:,i]=np.transpose(sp[i]*o)
		slitfunc_out[:,i]=np.transpose(o)
	
	if (uncert ==True) and (im_output ==True):
		return(sp,unc,im_out,slitfunc_out)#return unc perhaps as optional input parameter?
	elif uncert==True:
		return(sp,unc,slitfunc_out)
	elif im_output==True:
		return(sp,im_out,slitfunc_out)
	else: return(sp,slitfunc_out)
	
def sci_sp_func(order,ycen,osample,lamb_sp,lambda_sf,use_mask=0,noise=5.85, bkgd=0):
	if bkgd.shape!=order.shape:
		bkgd=np.zeros_like(order)
	#Tu mamy wczytany gotowy rzad w tablice "order"
	#Obtain dimensions of the order array
	nrow,ncol=order.shape
    #noise=0.0 #Noise should be asked as an input parameter above with default=0		  
	n=(nrow+1)*osample+1
	#test=np.zeros((ncol,2))
	#If no mask
	if (use_mask.shape!=order.shape):
		print'Extraction - Mask not supplied - generating a simple one for you'
    	#tworzymy maske, jako ze nie jest zdefiniowana
    	#create a mask, as it is not defined
		mask=np.ones_like(order)
		#j= np.where(order > 39999) #assumes divided by flat??
		mask[np.where(order > 39999)]=0.0
	else:
		#print 'Extraction - Using supplied mask'
		mask=np.copy(use_mask)
	#Sprawdzam, jak jest zdefiniowane lamb_sf
	#Checking for a defined lamb_sf
	if (lambda_sf==0.): 
		lambda_sf=0.1
	y=np.arange(n)/float(osample)-1.
	bklind=np.arange(osample+1)+n*osample
	oind=np.arange(osample+1)*(osample+2)
	olind=oind[0:osample+1]
	for m in range(osample+1, 2*osample+1):
		mm=m-osample
		bklind=np.append(bklind, np.arange(osample+1-mm)+n*m)
		olind=np.append(olind, oind[0:osample-mm+1]+mm)
	#Teraz idzie zgadywane widmo - na modle REDUCE
	#jako juz finalna postac jednowymiarowa
	#Now create spectrum - with REDUCE
	sp=np.repeat(0,ncol)
	sf=np.sum(order*mask,axis=1)
	#Konstrukcja pierwszego oszcowaniea slit function i widma
	#Liczenie mediany z piecioelementowych slotow - wedle procedur w reduce
	#Construction of the first slit function and spectrum
	#Counting median widths - according to the REDUCE procedure
	#Szacujemy pierwsza postac slit function
	#We estimate the first part of the slit function
	sf_med=np.arange(sf.shape[0]-4)
	for i in range(2, sf.shape[0]-2):
		sf_med[i-2]=np.median(sf[i-2:i+3])
	sf[2:sf.shape[0]-2]=sf_med
	sf=sf/np.sum(sf)
	#Szacujemy pierwsza postac widma
	#We estimate the first character of the spectrum
	sp=np.sum((order*mask)*(np.outer(sf,np.repeat(1,ncol))),axis=0)
	sp_med=np.arange(sp.shape[0]-4)
	for i in range(2, sp.shape[0]-2):# use scipy smoothing kernel later
		sp_med[i-2]=np.median(sp[i-2:i+3])
	sp[2:sp.shape[0]-2]=sp_med
	sp=sp/np.sum(sp)*np.sum(order*mask)#these sums were calculated prior...
	dev = np.sqrt(np.sum(mask*(order-np.outer(sf, sp))**2)/np.sum(mask))
	#j= np.where(abs(order-np.outer(sf, sp)) > 3.*dev)
	mask[np.where(abs(order-np.outer(sf, sp)) > 3.*dev)]=0.0
	#Wyznaczemie wagi
	#Calculate weights
	weight=1./np.float64(osample)

########################################################################
	for iter in range(1,25):
	#if iter == 2: gold = np.copy(sp)
	#teraz budujemy macierz, gdzie omega bedzie na przekatnej
	#build a matrix, where Omega will be the diagonal
		Akl=np.zeros((2*osample+1,n))
		Bl=np.zeros((1,n))
		omega=np.repeat(weight,osample+1)
		for i in range(0, ncol):
			#Tworzenie tablic wag, nie wymaga czytania rzedu
			#Creating arrays by weight, does not require reading trace
			omega=np.repeat(weight,osample+1)
			yy=y+ycen[i]
			ind=np.where((yy>=0.0) & (yy<1.))[0]
			i1=ind[0]
			i2=ind[-1]
			omega[0]=yy[i1]#Zmieniamy pierwsza wage w obrebie piksela
						   #Change the first wage within the pixel
			omega[-1]=1.-yy[i2]#Zmieniamy ostatnia wage w obrebie piksela
							   #Change the last wage within the pixel        
			#Ostateczna postac macierzy omega dla pojedynczego piksela
			#The final figure of omega matrix for each pixel
			o=np.outer(omega,omega) #najpierw transponujemy, bo w idlu jest inna notacja wymiarow macierzy (kolumna, wers)
									#first transpose, because idlu is another notation dimensional matrix (column, line)
			o[osample,osample]=o[osample,osample]+o[0,0] #zrobione wedle tego, co jest w reduce
														 #done according to what is in the reduce        
			bkl=np.zeros((2*osample+1,n))
			omega_t=np.reshape(o, o.shape[0]*o.shape[1])
			oo= omega_t[olind]
			for l in range(0, nrow):
				bkl_temp=np.reshape(bkl, bkl.shape[0]*bkl.shape[1])
				t=l*osample+bklind+i1
				bkl_temp[t]=oo*mask[l,i]
			bkl=np.reshape(bkl_temp,(2*osample+1,n))
			oo=o[osample, osample]
			for l in range(1, nrow):
				bkl[osample,l*osample+i1]=oo*mask[l,i]
			bkl[osample,nrow*osample+i1]=omega[osample]**2*mask[nrow-1,i]
			for m in range (0,osample):
				bkl[m,(osample-m):(n)]=bkl[2*osample-m,0:(n-osample+m)]
			Akl=Akl+(sp[i]**2)*bkl
		#Tu teoretycznie mamy Akl jak z artykulu o reduce rownianie 10
		#powinno byc poprawnie powyzej
		#Here we theoretically Akl article about how to reduce a par 10
		#should be properly above	
		#Teraz robimy macierz Bkl, ktora w artykule jest zapisana jako macierz R
		#Now we Bkl matrix, which the article has written as matrix R_k
			o=np.zeros((1,n))
			for l in range (0, nrow):
				o[0,l*osample+i1:l*osample+i1+osample+1]=order[l,i]*weight*mask[l,i]
			for l in range (1, nrow):
				o[0,l*osample+i1]=order[l-1,i]*omega[osample]*mask[l-1,i]+order[l,i]*omega[0]*mask[l,i]
			o[0,i1]=order[0,i]*omega[0]*mask[0,i]
			o[0,nrow*osample+i1]=order[nrow-1,i]*omega[osample]*mask[nrow-1,i]
			Bl=Bl+sp[i]*o
		#Koniec petli po i
		#mamy prawa strone rowania 8, zdefiniowana rownaniem 1
		#The end of the loop, and we have eight right side of the equation, defined equation, one
		tab=np.zeros((n,2))
		#definiujemy czynnik lambda
		lamda=lambda_sf*np.sum(Akl[osample,:])/n
		lambda_tab=np.zeros((1,n))
		for elem in range(0,n):
			lambda_tab[0,elem]=lamda
			#wkladamy czynnik lamda do tablicy Akl
			#rownowazne tablicy Bjk w pracy o reduce
			# put our factor lamda Akl equivalent to an array of array Bjk reduce the work of		
		Akl[osample,0]=Akl[osample,0]+lambda_tab[0,0]
		Akl[osample,n-1]=Akl[osample,n-1]+lambda_tab[0,n-1]
		Akl[osample,1:n-1]=Akl[osample,1:n-1]+2.*lambda_tab[0,1:n-1]
		Akl[osample+1,0:n-1]=Akl[osample+1,0:n-1]-lambda_tab[0,0:n-1]
		Akl[osample-1,1:n]=Akl[osample-1,1:n]-lambda_tab[0,1:n]
		Bl=Bl.T
		x = solve_banded((osample,osample), Akl, Bl, overwrite_ab=True, overwrite_b=True)
		ind0=[np.where(x<0)]
		x[ind0]=0.0
		sf=x/np.sum(x)*osample
		r=np.repeat(0.,sp.shape[0])
		sp_old=np.copy(sp)
		dev_new=0.0
		for i in range(0, ncol):
			omega=np.repeat(weight,osample)
			yy=y+ycen[i]
			ind1=np.where((yy>=0.0) & (yy<nrow))[0]
			i1=ind1[0]
			i2=ind1[-1]
			omega[0]=yy[i1]
			ssf=np.reshape(sf[i1:i2+1],(nrow, osample))
			o=np.dot(ssf,omega)
			#figures=True #plot slit funcs in iteration
			#if figures==True:
			#	if i>5:
			#		plt.plot(ssf)
			#		plt.show()
			yyy=nrow-yy[i2]
			o[0:nrow-1]=o[0:nrow-1]+ssf[1:nrow,0]*yyy
			o[nrow-1]=o[nrow-1]+sf[i2]*yyy
			r[i]=np.dot((order[:,i]*mask[:,i]),o)
			sp[i] = np.sum( o**2 * mask[:,i])
			if (iter > 1):
				norm=r[i]/sp[i]
				j= np.where(abs(order[:,i]-np.transpose(norm*o))>7.*dev)		
				mask[j,i]=0.0			
		
				dev_new=dev_new+np.sum(mask[:,i]*(order[:,i]-np.transpose(norm*o))**2)
		if (iter >1 ):
			dev=np.sqrt(noise**2+dev_new/np.sum(mask))
		if (lamb_sp != 0.0):
			lamda=lamb_sp*np.sum(sp)/ncol
			ab=np.zeros((3,ncol))
			ab[0,1:]=-lamda
			ab[2,:-1]=-lamda
			ab[1,0]=lamda+1.
			ab[1,-1]=lamda+1.
			ab[1,1:-1]=2.*lamda+1.
			sp=solve_banded((1,1), ab, r/sp, overwrite_ab=False, overwrite_b=False)
		else:
			sp = r/sp
		if ((abs(sp-sp_old)/sp.max()).max()<0.00001):
			break
	
	jbad=np.array(0,dtype=np.int64)
	unc=np.repeat(0.,ncol)
	im_out=np.zeros_like((order))
	var=np.repeat(0.,ncol)
	
	for i in range(0, ncol):
		omega=np.repeat(weight,osample)
		yy=y+ycen[i]
		ind1=np.where((yy>=0.0) & (yy<nrow))[0]
		i1=ind1[0]
		i2=ind1[-1]
		omega[0]=yy[i1]
		ssf=np.reshape(sf[i1:i2+1],(nrow, osample))
		o=np.dot(ssf,omega)		
		yyy=nrow-yy[i2]
		o[0:nrow-1]=o[0:nrow-1]+ssf[1:nrow,0]*yyy
		o[nrow-1]=o[nrow-1]+sf[i2]*yyy
		j = np.where((abs(order[:,i]-np.transpose(sp[i]*o))).flatten()<5*dev)
		b = np.where((abs(order[:,i]-np.transpose(sp[i]*o))).flatten()>=5*dev)
		nj=sp[j].shape[0]
		#done in three steps iteratively b are outliers
		#data 
		if (nj< nrow):
			jbad=np.append(jbad, nrow*i+b[0])
		if (nj>2):
			ss=np.sum((order[j,i]-sp[i]*o[j])**2)
			xx=np.sum((o[j]-np.mean(o[j]))**2)*(nj-2)
			unc[i]=ss/xx
		else:
			unc[i]=0.0
		#replace this unc with better
		im_out[:,i]=np.transpose(sp[i]*o)
		var[i]=sp[i]*(np.sum((im_out[:,i]+bkgd[:,i]+(np.ones_like(bkgd[:,i])*(noise**2))))/np.sum((im_out[:,i]*mask[:,i])))


	return sp, var
			
def extract_and_normalise_flat():
	print "Normalise Flat: Beginning process to find blaze functions and pixel sensitivity mask."
	#assumes premade scatter, orderlocations, and masterflat
	scatter=findscatter(file='radius_masterflat.fits')
	flat=pyfits.open('radius_masterflat.fits')
	
	try:
		flat_tab=flat[0].data*flat[0].header['RO_GAIN']
		scatter=scatter*flat[0].header['RO_GAIN']
	except:
		flat_tab=flat[0].data*2.85
		scatter=scatter*2.85
		
	try: noise=header['RO_NOISE']
	except: noise=5.85
	
	a,b=flat_tab.shape
	if b>a:
		flat_tab=np.transpose(flat_tab)
	a,b=flat_tab.shape
	xrange=np.arange(0,a,1)
	yrange=np.arange(0,b,1)
	
	if flat_tab.shape[0]!=scatter.shape[0]:
		scatter=scatter.T
	
	data=flat_tab-scatter

	orderparameters=np.load('order_definition_parameters.npy')
	#make array of ordertraces: (dispersionlength ,ordernumber)
	fitted_orders=np.zeros(len(xrange))
	#fitted_orders=np.zeros(([len(orderparameters),len(xrange)]))
	for ord, item in enumerate(orderparameters):
		if orderparameters[ord]!=[]:
			coeffs=orderparameters[ord]
			#print ord, coeffs
			yyy=np.polyval(coeffs,xrange)
			if ord==0:
				fitted_orders=yyy
			else:
				fitted_orders=np.vstack((fitted_orders,yyy))
	fitted_orders.sort(axis=0)        
	cen=fitted_orders.T
	
	cen=cen[:,3:-2] #skip first and last three orders
	
	#smoothing step in dispersion direction 15-20 px or spline before feed in to this
	flat_tab_mod=data
	#windowsize=12
	#scipy convolve
	#kern=np.ones(windowsize)/windowsize
	for y in yrange:
		#flat_tab_mod[:,y]=filters.convolve(data[:,y],kern,mode='nearest') - seemingly unneeded
		flat_tab_mod[:,y]=filters.median_filter(flat_tab_mod[:,y],25)
	
	width=21
	normed_flat=np.zeros(data.shape)
	slitfuncmodel=np.zeros(data.shape)
	a,b=cen.shape
	blazefunctions=np.zeros(cen.shape)
	uncertainties=np.zeros(cen.shape)
	for ord in np.arange(0,b,1):
		i=0.
		ycen=cen[:,ord] #15th order only
		for elem in cen[:,ord]:
			elem=int(elem)
			if (i==0):
				order=np.transpose(flat_tab_mod[i,elem-width:elem+width+1])
				i=i+1
			else: #elif
				if i<a:
					temp=np.transpose(flat_tab_mod[i,elem-width:elem+width+1])
					order=np.column_stack((order,temp))
					i=i+1
		osample=10 
		lambda_sf=25.0 
		lambda_sp=0.0 
		use_mask=0 #Else supply a mask - maybe this should iterate 1st pass make mask, repeat
		#if (np.isfinite(order).all() and not np.isnan(order).all())==True:
		#	print 'Normalise Flat: Data is good for order %s.' %(ord+1)
		ycen=np.modf(cen[:,ord])[0]
		print "Normalise Flat: Order %s being calculated."%(ord+1)
		blazefunc,uncert,reconstruction,slitfunc=flat_sp_func(order, ycen,osample,lambda_sp, lambda_sf,use_mask=0,uncert=True,im_output=True,noise=noise,slitfunc=True)
		for x in xrange:
			elem=int(cen[x,ord])
			normed_flat[x,elem-width:elem+width+1]=reconstruction[:,x]
			slitfuncmodel[x,elem-width:elem+width+1]=slitfunc[:,x]
		uncertainties[:,ord]=uncert
	mask=np.ones(normed_flat.shape,dtype='bool')
	temp=np.zeros(normed_flat.shape)
	for x in xrange:
		temp[x,:]=maxfilt(normed_flat[x,:], 25, mode='reflect')
	temp[temp<0.000001]=0.000001
	temp=normed_flat/temp
	mask[temp<0.25]=0.0#0.8
	normed_flat=(data+scatter)/(normed_flat+scatter) #now normed_flat is the px sens mask
	mask[normed_flat<0.25]=0.0#0.8
	normed_flat[mask == 0.0]=1.0
	var=np.zeros(cen.shape)
	#blazefuncs=True
	print "Normalise Flat: Now extracting blaze functions"
	blazefunctions=np.zeros(cen.shape)
	normed_data=data/normed_flat
	for ord in np.arange(0,b,1):
		i=0.
		ycen=cen[:,ord] #15th order only
		for elem in cen[:,ord]:
			elem=int(elem)
			if (i==0):
				order=np.transpose(normed_data[i,elem-width:elem+width+1])
				ordmask=np.transpose(mask[i,elem-width:elem+width+1])
				ordscatter=np.transpose(scatter[i,elem-width:elem+width+1])
				i=i+1
			else: #elif
				if i<a:
					temp=np.transpose(normed_data[i,elem-width:elem+width+1])
					temp_mask=np.transpose(mask[i,elem-width:elem+width+1])
					temp_scatter=np.transpose(scatter[i,elem-width:elem+width+1])
					order=np.column_stack((order,temp))
					ordmask=np.column_stack((ordmask,temp_mask))
					ordscatter=np.column_stack((ordscatter,temp_scatter))
					
					i=i+1
		osample=10 
		lambda_sf=25.0 
		lambda_sp=0.0 
		#if (np.isfinite(order).all() and not np.isnan(order).all())==False:
		#	print 'Normalise Flat: Extracting blaze function - Data is bad for order %s.' %(ord+1)
		ycen=np.modf(cen[:,ord])[0] #used to be np.modf()[0] tried round
		print "Normalise Flat: Extracting blaze function - Order %s being calculated."%(ord+1)
		blazefunc,sf=flat_sp_func(order, ycen,osample,lambda_sp, lambda_sf,use_mask=ordmask,uncert=False,im_output=False,noise=noise)
		blazefunctions[:,ord]=blazefunc
		#use blazefunc,sf to derive uncert with readnoise and background.

		for ind, i in enumerate(blazefunc):
			var[ind,ord]=blazefunc[ind]*(np.sum((sf[:,ind]*blazefunc[ind]+ordscatter[:,ind]+(np.ones_like(ordscatter[:,ind])*(noise**2))))/np.sum((sf[:,ind]*ordmask[:,ind]*blazefunc[ind])))

	print "Normalise Flat: Done with blaze functions"
	#np.save('uncert',uncertainties)
	np.save('blaze_functions', blazefunctions)
	np.save('px_mask', mask)
	np.save('normed_flat', normed_flat)
	np.save('slitfuncs',slitfuncmodel)
	#plot blazefunctions
	#figures=True
	figures=False
	if figures is True:
		plt.subplot(2,2,1)
		for i in np.arange(1,blazefunctions.shape[1]):
			plt.plot(blazefunctions[:,i])
		plt.xlim(0,2746)
		plt.subplot(2,2,4)
		plt.imshow(normed_flat.T)
		plt.subplot(2,2,2)
		plt.imshow(data.T)
		plt.subplot(2,2,3)
		plt.imshow(mask.T)
		plt.show()	
	
	print"Normalise Flat: Process completed, sorry it took so long."
	return blazefunctions,var,normed_flat,mask,slitfuncmodel
		
def extract_arc(input_file='17feb11901.fits'):
	#happens after badmask creation in normflat.
	#probably should polynomial background too.
	#as per flat extraction but a mere summing in the spatial direction 
	#- need no real fine tuning to this as it's more location than intensity specific
	#good enough to px scales at least.
	hdulist=pyfits.open(input_file)
	header=hdulist[0].header
	try:
		gain=header['RO_GAIN']
	except:
		gain=2.85
	try:
		read_noise=header['RO_NOISE']
	except:
		read_noise=5.25
	data=gain*hdulist[0].data

	a,b=data.shape
	if b>a:
		data=data.T
		a,b=data.shape
	
	xrange=np.arange(0,a,1)
	yrange=np.arange(0,b,1)
	orderparameters=np.load('order_definition_parameters.npy')
	#fitted_orders= trace_order_params(xrange,orderparameters)
	
	#make array of ordertraces: (dispersionlength ,ordernumber)
	fitted_orders=np.zeros(len(xrange))
	#fitted_orders=np.zeros(([len(orderparameters),len(xrange)]))
	for ord, item in enumerate(orderparameters):
		if orderparameters[ord]!=[]:
			coeffs=orderparameters[ord]
			#print ord, coeffs
			yyy=np.polyval(coeffs,xrange)
			if ord==0:
				fitted_orders=yyy
			else:
				fitted_orders=np.vstack((fitted_orders,yyy))
	fitted_orders.sort(axis=0)        
	cen=fitted_orders.T
	cen=cen[:,3:-2] #skip first and last three orders
	#print 'finding scattered light'
	background = findscatter(file=input_file,nsteps = 100,poly_order_x = 6,poly_order_y = 4,figures=False, method='single')
	px_mask=np.load('px_mask.npy')
	try: #from flat field extraction
		slitfuncs=np.load('slitfuncs.npy')
		got_slitfuncs=True
	except:
		got_slitfuncs=False
	
	if a!=slitfuncs.shape[0]:
		slitfuncs=slitfuncs.T
	if a!=background.shape[0]:
		background=background.T
	if a!=px_mask.shape[0]:
		px_mask=px_mask.T
	
	data=(data-background)*px_mask #background subtraction
	slitfuncs=slitfuncs*px_mask
	flat_tab_mod=data
	#windowsize=12
	#scipy convolve
	#kern=np.ones(windowsize)/windowsize
	
	width=13

	a,b=cen.shape
	ext_arc=np.zeros(cen.shape)
	ext_arc_basic=np.zeros(cen.shape)
	unc=np.zeros(cen.shape)
	variance=np.zeros(cen.shape)
	var=np.zeros(cen.shape)
	for ord in np.arange(0,b,1):
		i=0.
		ycen=cen[:,ord] #15th order only
		for elem in cen[:,ord]:
			elem=np.round(elem)
			if (i==0):
				order=np.transpose(flat_tab_mod[i,elem-width:elem+width+1])
				if got_slitfuncs==True:
					sfunc=np.transpose(slitfuncs[i,elem-width:elem+width+1])
					ord_mask=np.transpose(px_mask[i,elem-width:elem+width+1])
				bgd=np.transpose(background[i,elem-width:elem+width+1])
				i+=1
			else: #elif
				if i<a:
					temp=np.transpose(flat_tab_mod[i,elem-width:elem+width+1])
					order=np.column_stack((order,temp))
					if got_slitfuncs==True:
						tempsfunc=np.transpose(slitfuncs[i,elem-width:elem+width+1])
						sfunc=np.column_stack((sfunc,tempsfunc))
						temp_ord_mask=np.transpose(px_mask[i,elem-width:elem+width+1])
						ord_mask=np.column_stack((ord_mask,temp_ord_mask))
					tempbgd=np.transpose(background[i,elem-width:elem+width+1])
					bgd=np.column_stack((bgd,tempbgd))
					i+=1

		order[order<0.0]=0.0
		sfunc[sfunc<0.0]=0.0
	
		for column in xrange: #can put better uncert alg in loop in extraction
			colmask=ord_mask[:,column]
			xxx=sfunc[:,column]
			xxx=xxx[np.where(colmask==True)]
			colback=bgd[:,column]
			#colback=colback[np.where(colmask==True)]
			colorder=order[:,column]
			colorder=colorder[np.where(colmask==True)]
			ext_arc[column,ord]=least_squares_solver(xxx[:,None],colorder)

			#var is the variance array
			var[column,ord]=ext_arc[column,ord]*np.sum((sfunc[:,column]*ext_arc[column,ord]+colback+(read_noise**2)))/np.sum((xxx*ext_arc[column,ord]))
			#where colmask==True
			#aa=xxx[:,None]
			#aa=aa[np.where(colmask==True)]
			#bb=np.abs((order[:,column]-read_noise**2)) #order=data-background
			#bb=bb[np.where(colmask==True)]
			#variance[column,ord]=least_squares_solver(aa,bb)
			
			
		print 'arc order %s acquired' %ord
		#sum in spatial direction for arc?
		#need to rewrite p&V to pull out sp profiles prior to reconstruction
		#option - try using arc spatial profiles to extract arc spectra...
		#more time consuming but better
		# they're saved as slitfuncs.npy
		#perhaps not so time consuming as I can use the compiled linear solver
		#for order in orders, for slice in x, solve I*slitfunction[slice]*mask=order[slice]*mask
		#mask will set I*0=0 - will not weight the soln by those points
		#but will need catch if vals.all() == 0
		
		### Simple extraction - can be commented out once satisfied with ext+unc
	#	try: # this gets average value spatially
	#		ext_arc_basic[:,ord]=np.sum(order,axis=0)/(order!=0).sum(0)
	#	except: print 'failed simple order extraction for %s'%ord #who cares really
	#plotting loop
	
	#plt.subplot(2,1,1)
	#plt.plot(ext_arc[:,20])
	#plt.subplot(2,1,2)
	#plt.plot(var[:,20]**0.5,color='red')
	#plt.show()

	#one should note from the plot that the differences between the spatial profile extraction
	#and the simple non-zero mean method are *very small
	#the computation overhead is small also (compiled C based linear solver), so the spatial profile
	#extraction method should be used. There is also some slight oddity in the 
	#stepwise nature of the mean solution due to the number of pixels available to the non-zero method.
	#this makes some step function effects in the extraction vs the sp profile method.

	return ext_arc,var

def extract_science_or_I2flat(input_file='17feb12035.fits'):
	#get ordlocs from file
	#calc bkgd from ordlocs, input_file
	#get noise,gain from input_file
	#get mask from saved file or extract_and_normalise_flat
	#extract - retain sp, ext
	#calc uncert form ext,sp,mask,bkgd,noise
	hdulist=pyfits.open(input_file)
	header=hdulist[0].header
	try:
		gain=header['RO_GAIN']
	except:
		gain=2.85
	try:
		read_noise=header['RO_NOISE']
	except:
		read_noise=5.25
		
	data=gain*hdulist[0].data

	a,b=data.shape
	if b>a:
		data=data.T
		a,b=data.shape
	
	xrange=np.arange(0,a,1)
	yrange=np.arange(0,b,1)
	orderparameters=np.load('order_definition_parameters.npy')
	#fitted_orders= trace_order_params(xrange,orderparameters)
	
	#make array of ordertraces: (dispersionlength ,ordernumber)
	fitted_orders=np.zeros(len(xrange))
	#fitted_orders=np.zeros(([len(orderparameters),len(xrange)]))
	for ord, item in enumerate(orderparameters):
		if orderparameters[ord]!=[]:
			coeffs=orderparameters[ord]
			#print ord, coeffs
			yyy=np.polyval(coeffs,xrange)
			if ord==0:
				fitted_orders=yyy
			else:
				fitted_orders=np.vstack((fitted_orders,yyy))
	fitted_orders.sort(axis=0)        
	cen=fitted_orders.T
	cen=cen[:,3:-2] #skip first and last three orders
	a,b=cen.shape
	#print 'finding scattered light'
	background = findscatter(file=input_file,nsteps = 100,poly_order_x = 6,poly_order_y = 4,figures=False, method='single')
	px_mask=np.load('px_mask.npy')
	
	if data.shape[0]!=background.shape[0]:
		background=background.T
	data=data-background#care here!!!!!
	
	#have to deal with normalised flat division too :/
	#what's the deal with this vs blazefunc correction? does it fix it?
	normed_flat=np.load('normed_flat.npy')
	if normed_flat.shape!=data.shape:
		normed_flat=normed_flat.T
	data=data/normed_flat
	osample=10 
	lambda_sf=25.0 
	lambda_sp=0.0 
	width=8
	print "Exctraction: Now extracting spectrum and uncertainty"
	extracted_spectrum=np.zeros(cen.shape)
	spectrum_variance=np.zeros(cen.shape)
	for ord in np.arange(0,b,1):
		i=0.
		ycen=cen[:,ord] 
		for elem in cen[:,ord]:
			elem=int(elem)
			if (i==0):
				order=np.transpose(data[i,elem-width:elem+width+1])
				ordmask=np.transpose(px_mask[i,elem-width:elem+width+1])
				ordscatter=np.transpose(background[i,elem-width:elem+width+1])
				i=i+1
			else: #elif
				if i<a:
					temp=np.transpose(data[i,elem-width:elem+width+1])
					temp_mask=np.transpose(px_mask[i,elem-width:elem+width+1])
					temp_scatter=np.transpose(background[i,elem-width:elem+width+1])
					order=np.column_stack((order,temp))
					ordmask=np.column_stack((ordmask,temp_mask))
					ordscatter=np.column_stack((ordscatter,temp_scatter))	
					i=i+1
			osample=10 
			lambda_sf=25.0 
			lambda_sp=0.0 
		if (np.isfinite(order).all() and not np.isnan(order).all())==False:
			print 'Extracting %s - Data bad for order %s.' %(input_file,ord+1)
		ycen=np.modf(cen[:,ord])[0] #used to be np.modf()[0] tried round
		print "Extracting %s - Order %s being calculated."%(input_file,ord+1)
		ext,var=sci_sp_func(order,ycen,osample=10,lamb_sp=0.0,lambda_sf=25.0,use_mask=ordmask,noise=read_noise,bkgd=ordscatter)
		extracted_spectrum[:,ord]=ext
		spectrum_variance[:,ord]=var
	#define some spatial extent for order extraction
	#call p&V ext method, output ext, unc
	#sp for plotting
	figs=False
	#I have some substantial stuff around to do with the formatting of the output.
	if figs==True:
		plt.subplot(2,2,1)
		plt.plot(extracted_spectrum[:,17])
		plt.subplot(2,2,3)
		plt.plot(spectrum_variance[:,17]**0.5,color='r')
		plt.subplot(2,2,2)
		plt.plot(extracted_spectrum[:,27])
		plt.subplot(2,2,4)
		plt.plot(spectrum_variance[:,27]**0.5,color='r')
		plt.show()
	return extracted_spectrum,spectrum_variance

#example usage
#blazefunctions,var,normed_flat,mask,slitfuncmodel=extract_and_normalise_flat()
#sp,unc=extract_science_or_I2flat(input_file='....')

#def wavesol():
#to do prior to functional_blazes() fitting

#def functional_blazes():
#blazefunctions=np.load('blaze_functions.npy')
#fit generalised functional form to the blazes to deal with fringing
#need wavelength solution to finish this
#(k/2)*(  ((c/(k+0.17))/wavelength) -1) = nu where k = order number and c is some const ~3959000
#as you see - needs wavelength solution
#for order in orders do:
	
