import os
import numpy as np
from time import strftime as strftime #for reduction time/date only
from astropy.io import fits as pyfits
from astropy.io import ascii
import pickle
from astropy.time import Time 
from datetime import datetime

#to run this stuff
#reducebysubfolder(os.getcwd())
#should get it going

debug=False
figures=False
	
def loadfits(Currentfile):
	hdulist = pyfits.open(Currentfile, ignore_missing_end= True)#,memmap=False
	return hdulist

def save_reduced_data(input_file,sp,unc,wave=None):
	print 'Saving extraction as %s'%(os.path.splitext(input_file)[0]+'_reduced.fits')
	#load up file to grab header to append reduction information to
	hdulist=loadfits(input_file)
	head=hdulist[0].header
	head['ORDERS'] = (sp.shape[1], 'Number of orders reduced')
	#append header field 'how many orders',len(ord)
	head['REDUTIME'] = (strftime("%c"), 'When reduced')
	#append header field 'when reduced', time.strftime("%c")
	head['comment'] = 'data saved as (ext,unc,wave) for each order'
	
	if wave==None:
		wave=np.arange(sp.shape[0])
	data=[[]]*(sp.shape[1]+1)
	for i in np.arange(sp.shape[1]):
		data=np.vstack((sp[:,i],unc[:,i],wave[i,:][::-1]))
		if i==0:
			head['ORDER']=((i+1),'Order number')
			pyfits.writeto(os.path.splitext(input_file)[0]+'_reduced.fits', data,head, clobber=True)
		else:
			head['ORDER']=((i+1),'Order number')
			pyfits.append(os.path.splitext(input_file)[0]+'_reduced.fits', data,head)
	#save file as original file + _reduced
	#ie. if aug008811.fits was the file - this becomes
	#aug008811_reduced.npy
	hdulist.close()

#string checking for Identify image method
def Anymatching(a,b):
	#print a,b
	#assumes a,b = list of strings , a string respectively
	#(used in header recognition)
	c=any(str(k) in str(b) for k in a)
	#print c
	return c
			
#use UI to get folder selector output - call this 'filepath' for prepare
def Prepare(filepath):
	#Tab1dialogtext.append("Finding fits files under selected folder")
	if debug == True:
		print "---------------------------------"
		print "   Getting directory structure"
		print "---------------------------------"				
	os.chdir(filepath)
	FITSpaths = open("FITSpaths.txt", "w")			
	for dirname, dirnames, filenames in os.walk('.'):
		# print path to all filenames if FITS.
		for filename in filenames:
			if '.fits' in filename:
				if 'radius_masterflat' not in filename and 'drt' not in filename and '.old' not in filename and 'master' not in filename and 'reduced' not in filename:	
					pathto = os.path.join(dirname,filename)
					if debug == True:
						print 'found %s' %pathto
						#Tab1dialogtext.append(pathto)
					FITSpaths.write(str(pathto) + os.linesep) 	
	FITSpaths.close()
	#Tab1dialogtext.append("Found files. Please identify image files.") 

#IdentifyImage loads the list made by Prepare and figures out what the thing is
def IdentifyImage():
	#reading header fields     
	filedata={}
	filedata['FILE']=[]
	keys = ['UTMJD','OBJECT', 'OBSTYPE', 'MEANRA', 'MEANDEC', 'INSTRUME', 'FOCALMOD', 'FIRMVSYS','REDUCEAS']
	for i in keys:
		filedata[i] = []
	with open('FITSpaths.txt', 'r') as f:
		global FITSfiles
		FITSfiles = [line.strip() for line in f] #make list of FITSpaths   
		if debug == True:
			print "-------------------------------"
			print "Getting header information"
			print "-------------------------------"
		for FITSfile in FITSfiles:
			PROBLEM=False
			Currentfile = FITSfile
			filedata['FILE'].append(Currentfile)
			if debug == True:
				print "Assessing files %.0f%% : (%s)." %(len(filedata['FILE'])*100/len(FITSfiles),Currentfile)
			#Tab1dialogtext.append("Assessing files %.0f%% : (%s)." %( len(filedata['FILE'])*100/len(FITSfiles),Currentfile) )
			hdulist=loadfits(Currentfile)
			oldheader=hdulist[0].header
			#try:
			#	hdulist = RemoveOverscan(currentfile=hdulist,savename=Currentfile)
			#except: pass
			try:
				for j in keys: #are all headers present?
					temp = hdulist[0].header[j]
				
				if j == 'REDUCEAS':
				#if hdulist[0].header['REDUCEAS'] in ['SKIP', '','[]']:
					FAIL=hdulist[0].header['NON_EXISTENT_HEADER_FIELD']
					#go to re-identify if skip
					
				for j in keys: #populate table
					filedata[j].append(hdulist[0].header[j])

				print'%s file already identified as %s' %(Currentfile, hdulist[0].header['REDUCEAS'])
			except:
				if hdulist[0].header['NAXIS'] == 0 or hdulist[0].header['NAXIS   '] == 0:
					METHOD = 'SKIP'
					print'Header shows no data - skipping'
					hdulist[0].header['REDUCEAS']=(METHOD,'Reduction method to use for frame')
					if hdulist[0].header != oldheader:
						print'Writing SKIP to header'
						hdulist.writeto(FITSfile,clobber=True)
					for j in keys:
						try:
							filedata[j].append(hdulist[0].header[j])
						except:
							filedata[j].append('null')
				elif Anymatching(['AAOMEGA-IFU','CYCLOPS','Cyclops','CYCLOPS2','Cyclops2','TAURUS','Taurus','Taurus  '],hdulist[0].header['INSTRUME']):
					METHOD = 'SKIP'
					print'image is from the %s instrument - will be skipped'%hdulist[0].header['INSTRUME']
					hdulist[0].header['REDUCEAS']=(METHOD,'Reduction method to use for frame')
					if hdulist[0].header != oldheader:
						print'Writing SKIP to header'
						hdulist.writeto(FITSfile,clobber=True)
					for j in keys:
						try:
							filedata[j].append(hdulist[0].header[j])
						except:
							filedata[j].append('null')
				elif Anymatching(['nulnd','Nulnd','NuInd','nuind','test','Test','tests','RUN','run','PellicleTests','focus','FOCUS','Focus','dummy','DUMMY'],hdulist[0].header['OBJECT']):
					METHOD = 'SKIP'
					print'image is a %s - will be skipped'%hdulist[0].header['INSTRUME']
					hdulist[0].header['REDUCEAS']=(METHOD,'Reduction method to use for frame')
					if hdulist[0].header != oldheader:
						print'Writing SKIP to header'
						hdulist.writeto(FITSfile,clobber=True)
					for j in keys:
						try:
							filedata[j].append(hdulist[0].header[j])
						except:
							filedata[j].append('null')			
				else:
					for j in keys:
						try:
							if j=='REDUCEAS':
								FAIL=hdulist[0].header['NON_EXISTENT_HEADER_FIELD']
							temp = hdulist[0].header[j]
							#if header is here then
							filedata[j].append(temp)
							#the header keyword doen't exist this happens for a variety of reasons
						except:
							if j in ['FIRMVSYS']:
								try:
									temp = hdulist[0].header['CCD_VER']
									if debug == True:
										print "  FIRMVSYS missing - used CCD_VER"
									#Tab1dialogtext.append("FIRMVSYS missing - used CCD_VER")
									hdulist[0].header[j]=(temp,'Duplicated from CCD_VER')
								except:
									try:
										temp = hdulist[0].header['DETECTOR']
										if debug == True:
											print "%s missing - used DETECTOR"%j
										#Tab1dialogtext.append("FIRMVSYS, CCD_VER missing - used DETECTOR")
										hdulist[0].header[j]=(temp,'Duplicated from DETECTOR')
									except:
										temp='null'
										PROBLEM=True
								filedata[j].append(temp)
							elif j in ['UTMJD']:
								print'UTMJD not found'
								try:
									#construct MJD from UTSTART + 1/2 * TOTALEXP
									tempdate = hdulist[0].header['UTDATE']
									exptime = hdulist[0].header['TOTALEXP']
									exptime_days = exptime/60/60/24
									tempday = (datetime.strptime(tempdate, "%Y:%m:%d")).strftime('%Y-%m-%d')
									temptime = hdulist[0].header['UTSTART']
									times = tempday + " " + temptime
									t = Time(times, scale='utc')
									newmjd = t.mjd + exptime_days
									if debug == True:				
										print "  UTMJD missing - calculated new UTMJD from UTSTART and TOTALEXP = %r" % newmjd			
									#Tab1dialogtext.append("UTMJD missing - calculated new UTMJD from UTSTART and TOTALEXP = %r" %newmjd)
									hdulist[0].header[j]=(newmjd,'UTMJD calculated from UTSTART and TOTALEXP')
									print'UTMJD created'
								except KeyError: 
									try:
										#construct MJD from UTSTART & UTEND 
										tempdate = hdulist[0].header['UTDATE']
										exptime = hdulist[0]
										tempday = datetime.strptime(tempdate, "%Y:%m:%d").strftime('%Y-%m-%d')
										temptime = hdulist[0].header['UTSTART']
										times = tempday + " " + temptime
										t = Time(times, scale='utc')
										newmjdstart = t.mjd
										temptime = hdulist[0].header['UTEND']
										times = tempday + " " + temptime
										t = Time(times, scale='utc')
										newmjdend = t.mjd
										t=[newmjdstart, newmjdend]
										newmjd=np.mean(t)	
										if debug == True:				
											print "  UTMJD missing - calculated new UTMJD from UTSTART and UTEND = %r" %newmjd			
										#Tab1dialogtext.append("UTMJD missing - calculated new UTMJD form UTSTART and UTEND = %r" %newmjd)
										hdulist[0].header[j]=(newmjd,'UTMJD calculated from UTSTART and UTEND')
										print'UTMJD created'
									except KeyError:
										newmjd="null"
										if debug == True:				
											print "  UTMJD missing - unable to create one"	
										#Tab1dialogtext.append("UTMJD missing -unable to create one ")
										#Tab1dialogtext.append("Strongly suggest SKIP as this header is malformed")
								filedata[j].append(newmjd)
								
							elif j in ['MEANRA', 'MEANDEC']:
								try:
								#record tracking - likely parked at zenith
									temp = hdulist[0].header['TRACKING']
									if debug == True:
										print "  %s missing - used TRACKING" % j
										#Tab1dialogtext.append("No MEANRA, MEANDEC - used TRACKING header")
									hdulist[0].header[j]=(temp,'Duplicated from TRACKING')
									print'TRACKING used'
								except KeyError:
									if debug == True:	
										print "  %s missing - null entry" % j
									#Tab1dialogtext.append("No MEANRA, MEANDEC, nor TRACKING recorded")			
									#Tab1dialogtext.append("This is a problem if a stellar image")										
									temp='null'
									PROBLEM = True
								filedata[j].append(temp)
							elif j in ['FOCALMOD','OBSTYPE']:
								print '%s absent from header'%j
								#might be old
								if debug == True:
									print" %s missing"%j
								try:
									temp=hdulist[0].header['DETECTOR']									
									#if temp in ['CCD_2','MITLL2','TEKTRONIX','MITLL2A','MITLL2a','EEV2','EEV2    ','MITLL3','TEKTRONIX_5']:
									hdulist[0].header[j]=(temp,'Duplicated from DETECTOR')
									print 'detector is %s, used for %s header field' %temp,j
								except:
									try:
										temp=hdulist[0].header['INSTRUME']									
										#if temp in ['CCD_2','MITLL2','TEKTRONIX','MITLL2A','MITLL2a','EEV2','EEV2    ','MITLL3','TEKTRONIX_5']:
										hdulist[0].header[j]=(temp,'Duplicated from INSTRUME')
										print 'detector is %s, used for %s header field' %temp,j
									except:
										print 'detector not recognised, that is not ideal'
										temp='null'
										#PROBLEM=True
								filedata[j].append(temp)
							elif j !='REDUCEAS':#damned typo!!!!
								#problem file - go to user id mode
								if debug == True:
									print "  %s missing - null entry" % j			
								filedata[j].append('null')
								#Tab1dialogtext.append("Something is missing from the header of this file")
								#Tab1dialogtext.append("Waiting for user identification of this image type")
								PROBLEM = True
								print "%s not found in header of %s"%(j,Currentfile)
					#Choosing how reduction treats each frame. Note old (1998) spectra have no FOCALMOD, OBSTYPE - deal with it
					try:
						if  Anymatching(['bias', 'BIAS', 'Bias','BIAS_0001'],hdulist[0].header['OBJECT']) and hdulist[0].header['TOTALEXP']== 0:
							METHOD = 'BIAS' 
						elif Anymatching(['null','Null','NULL'],hdulist[0].header['UTMJD']):
							METHOD='SKIP'
						elif Anymatching(['FLAT', 'Flat', 'flat','wideflat','WIDEFLAT','Wdieflat','wdieflat','Flat sl=6arcsec'],hdulist[0].header['OBSTYPE'])and Anymatching(['CLEAR', 'Clear' ,'clear'],hdulist[0].header['FOCALMOD'])and Anymatching(['WIDEFLAT', 'WideFlat', 'wideflat', 'Wideflat','Flat sl=6arcsec','wide','wdie','WIDE','WDIE'],hdulist[0].header['OBJECT']):
							METHOD = 'WIDEFLAT'
						elif Anymatching(['iodine', 'IODINE', 'Iodine','I2','Iodine 0.5arcsec','Iodine  '],hdulist[0].header['OBJECT']) and Anymatching(['Iodine  ','iodine', 'IODINE', 'Iodine','I2','Iodine 0.5arcsec'],hdulist[0].header['FOCALMOD']) and Anymatching(['FLAT', 'Flat', 'flat'], hdulist[0].header['OBSTYPE']):
							METHOD = 'I2FLAT'
						elif Anymatching(['FLAT', 'Flat', 'flat','narrowflat','NARROWFLAT','narrow','Narrow','NARROW'],hdulist[0].header['OBSTYPE']) and Anymatching(['CLEAR', 'Clear' ,'clear'],hdulist[0].header['FOCALMOD'])and Anymatching(['NARROWFLAT', 'NarrowFlat', 'narrowflat', 'Narrowflat','Narrow','NARROW','narrow'],hdulist[0].header['OBJECT']) :
							METHOD = 'NARROWFLAT'
						elif Anymatching(['ARC', 'Arc', 'arc','THAR','thar','ThAr','Thar','ThAr0.5px','ThAr1.0px','ThAr 0.5pix+cell'],hdulist[0].header['OBJECT']) and Anymatching(['ThAr', 'THAR', 'Thorium', 'thar','ARC', 'Arc', 'arc','ThAr0.5px','ThAr1.0px','ThAr 0.5pix+cell'],hdulist[0].header['OBSTYPE']):
							METHOD = 'ARC' #problems identifying arc in str(arcsec)
						# many of these lack additional checks apart from just OBJECT - may need some stat measure and/or ECHGAMMA, ECHTHETA and != 31 l/mm ??.
						elif Anymatching(['WIDEFLAT', 'WideFlat', 'wideflat', 'Wideflat','wide','WIDE','Wide','FibFlat','SlitFlat','Slitflat','Wdieflat','wdieflat'],hdulist[0].header['OBJECT']) :
							METHOD = 'WIDEFLAT'
						elif Anymatching(['iodine', 'IODINE', 'Iodine','Iflat','I2flat','Iflat','IFLAT','I2','Iodine 0.5arcsec','Iodine  '],hdulist[0].header['OBJECT']):
							METHOD = 'I2FLAT'
						elif Anymatching(['NARROWFLAT', 'NarrowFlat', 'narrowflat', 'Narrowflat'],hdulist[0].header['OBJECT']):
							METHOD = 'NARROWFLAT'
						elif ((Anymatching(['iodine', 'IODINE', 'Iodine'],hdulist[0].header['FOCALMOD']) and hdulist[0].header['TOTALEXP']!= 0) or hdulist[0].header['OBJECT'].isdigit()) and not Anymatching(['flat','Flat','arc','ARC','Thar','THAR','quatrz','QUARTZ', 'RUN','run','Run','Test','test'],hdulist[0].header['OBJECT']):
							METHOD = 'SCIENCE'
						elif Anymatching(['DARK', 'Dark', 'dark'],hdulist[0].header['OBJECT']):
							METHOD = 'DARK'
							#print 'Goodness you found a dark image - how rare'
						elif Anymatching(['bias', 'BIAS', 'Bias','BIAS_0001'],hdulist[0].header['OBJECT']) :
							METHOD = 'BIAS' 
						elif Anymatching(['ARC', 'Arc', 'arc','THAR','thar','ThAr','Thar','ThAr0.5px','ThAr1.0px','ThAr 0.5pix+cell'],hdulist[0].header['OBJECT']) :
							METHOD = 'ARC'
						elif hdulist[0].header['TOTALEXP']!= 0 and Anymatching(['HD','hd','Hd','HR','hr','Hr'],hdulist[0].header['OBJECT']) :
							METHOD = 'SCIENCE' #RISKY!
						elif hdulist[0].header['OBJECT']=='IODINE':
							METHOD = 'I2FLAT'
						elif hdulist[0].header['OBJECT']=='THAR':
							METHOD = 'ARC'
						elif hdulist[0].header['OBJECT']=='WIDEFLAT':
							METHOD = 'WIDEFLAT'
						elif hdulist[0].header['OBJECT'].isdigit():
							METHOD = 'SCIENCE' 								
						else: METHOD='SKIP'
						hdulist[0].header['REDUCEAS']=(METHOD,'Reduction method to use for frame')
					except:
						PROBLEM = True
						print "Identification problem in %s" %Currentfile
						METHOD='SKIP'
						hdulist[0].header['REDUCEAS']=(METHOD,'Reduction method to use for frame')
			
					if PROBLEM ==False:
						if METHOD!='SCIENCE':
							print "%s file identified as %s" %(Currentfile, METHOD)
						else:
							print "%s file identified as %s with object %s" %(Currentfile, METHOD, hdulist[0].header['OBJECT'])
						#Tab1dialogtext.append("%s file identified as %s" %(Currentfile, METHOD))
			
					else:
						if debug == True:	
							Tab1plots(Currentfile)
							break
						else: 
							METHOD = 'SKIP'
							print "%s file not identified: to skip" %(Currentfile)
							#Tab1dialogtext.append("%s file not identified: %s" %(Currentfile, METHOD))
							hdulist[0].header['REDUCEAS']=(METHOD,'Reduction method to use for frame')
					
					if hdulist[0].header['REDUCEAS'] != oldheader['REDUCEAS']:
						print'Writing file changes'
						hdulist.writeto(FITSfile,clobber=True)
						#del hdulist #if memory woes
					print "Percent done: %s"%(len(filedata['FILE'])*100/len(FITSfiles))
					filedata['REDUCEAS'].append(METHOD)
					
					if len(filedata['REDUCEAS'])!=len(filedata['UTMJD']):
						for j in keys:
							print j,hdulist[0].header[j]
						print"***** problem with %s, reduction methods allocated not same as UTMJDs found"%Currentfile
		#np.save('data_array', filedata)
		ascii.write(filedata, "Inventoryspectra.dat")	
		np.save('filedata',filedata)
		return filedata

#master image methods methods
def make_master_bias(filedata):
	#filedata=np.load('filedata.npy')
	biasfiledata={} #preset structure
	keys = ['UTMJD','OBJECT', 'OBSTYPE', 'MEANRA', 'MEANDEC', 'INSTRUME', 'FOCALMOD', 'FIRMVSYS','REDUCEAS','FILE']
	for i in keys:
		biasfiledata[i] = []					
	#global filedata
	for j in range(len( filedata['REDUCEAS'] )):
		if filedata['REDUCEAS'][j] == 'BIAS':
			for k in keys:
				biasfiledata[k].append(filedata[k][j])		

	if len(biasfiledata)%2==0: 		#if even number of files
		np.delete(biasfiledata, 0, 0) 	#delete first entry to make number odd
	
	biasfiles=biasfiledata['FILE']	#this is a list of filenames
	if len(biasfiles)<=1:
		print"Error: Too few identified bias files"
		#Tab2dialogtext.append("Unable to make master bias: Too few identifed bias files!")
	else:	
		biasdata={}				#preset structure
		print "loading %.0f bias image arrays" % len(biasfiles)
		for f in biasfiles:			#for each filename
			biasdata[f] = pyfits.getdata(f)	#use filename as header and read data as array for each header
	
		print "making master bias image - this can take a little time"
		medianbias=np.median(biasdata.values(),axis=0)	#take median of all the data arrays
		print "done making master bias - saving and plotting for inspection"

		hdu=pyfits.PrimaryHDU(medianbias)
		hdu.header.add_comment("Master bias constructed from median of %.0f bias images" % len(biasfiles))
		meanut= np.mean(biasfiledata['UTMJD'])
		hdu.header['UTMJD']=(meanut,'mean UTMJD of raw bias images used')
		minut=min(biasfiledata['UTMJD'])
		maxut=max(biasfiledata['UTMJD'])
		hdu.header['MINUTMJD']=(minut,'minimum UTMJD of raw bias images used')
		hdu.header['MAXUTMJD']=(maxut,'maximum UTMJD of raw bias images used')
		
		headget=pyfits.getheader(biasfiles[0])
		try:
			hdu.header['FOCALMOD']=headget['FOCALMOD']
		except:
			hdu.header['FOCALMOD']=headget['OBSTYPE']
		
		hdulist = pyfits.HDUList([hdu])
		hdulist.writeto("radius_masterbias.fits", clobber=True)
		hdulist.close()
def make_master_flat(filedata):
	#filedata=np.load('filedata.npy')
	flatfiledata={} #preset structure

	keys = ['UTMJD','OBJECT', 'OBSTYPE', 'MEANRA', 'MEANDEC', 'INSTRUME', 'FOCALMOD', 'FIRMVSYS','REDUCEAS','FILE']
	for i in keys:
		flatfiledata[i] = []					
	#global filedata
	for j in range(len( filedata['REDUCEAS'] )):
		if filedata['REDUCEAS'][j] =='WIDEFLAT':
			for k in keys:
				flatfiledata[k].append(filedata[k][j])		
	
	if len(flatfiledata)%2==0: 		#if even number of files
		np.delete(flatfiledata, 0, 0) 	#delete first entry to make number odd
	
	
	flatfiles=flatfiledata['FILE']	#this is a list of filenames
	if len(flatfiles)<=1:
		print"Error: Too few identified flat files"
		#Tab2dialogtext.append("Unable to make master flat: Too few identifed flat files!")
	else:	
		flatdata={}				#preset structure
		print "loading %s flat image arrays" % len(flatfiles)
		for f in flatfiles:			#for each filename
			flatdata[f] = pyfits.getdata(f)	#use filename as header and read data as array for each header
	
		print "making master flat image - this can take a little time"
		medianflat=np.median(flatdata.values(),axis=0)	#take median of all the data arrays
		print "done making master flat - saving and plotting for inspection"

		hdu=pyfits.PrimaryHDU(medianflat)
		hdu.header.add_comment("Master flat constructed from median of %.0f flat images" % len(flatfiles))
		meanut= np.mean(flatfiledata['UTMJD'])
		hdu.header['UTMJD']=(meanut,'mean UTMJD of raw flat images used')
		minut=min(flatfiledata['UTMJD'])
		maxut=max(flatfiledata['UTMJD'])
		hdu.header['MINUTMJD']=(minut,'minimum UTMJD of raw flat images used')
		hdu.header['MAXUTMJD']=(maxut,'maximum UTMJD of raw flat images used')
	
		headget=pyfits.getheader(flatfiles[0])
		try:
			hdu.header['FOCALMOD']=headget['FOCALMOD']
		except:
			hdu.header['FOCALMOD']=headget['DETECTOR']
		#need gain and ro_noise too
		try:
			hdu.header['RO_GAIN']=headget['RO_GAIN']
		except: 
			try:
				hdu.header['RO_GAIN']=headget['GAIN']
			except: pass
		
		try:
			hdu.header['RO_NOISE']=headget['RO_NOISE']
		except: 
			try:
				hdu.header['RO_NOISE']=headget['NOISE']
			except: pass
					
		hdulist = pyfits.HDUList([hdu])
		hdulist.writeto("radius_masterflat.fits", clobber=True)
		hdulist.close()

###sequential reduction
def reducebysubfolder(filepath):
	initial_path= os.getcwd()
	for f in os.listdir(filepath):
		print "Current working directory is %s" % initial_path
		child=os.path.join(filepath,f)
		if os.path.isdir(child): 
			print f
			print'*** Beginning reduction of files within %s ***'%child
			Prepare(child)
			print'*** Identifying files ***'
			filedata=IdentifyImage()
			#filedata=np.load('filedata.npy')
			print'*** Making master templates ***'
			make_master_bias(filedata=filedata) #likely to fail as very few bias images
			make_master_flat(filedata=filedata) #uses filelist and 'REDUCEAS' categorisation
			trace_orders() #uses saved master flat, saves orderparameters to file
			print'*** Reducing Master Flat ***'
			blazefunctions,var,normed_flat,mask,slitfuncmodel=extract_and_normalise_flat()
			#save_reduced_data('radius_masterflat.fits',blazefunctions,var)
			print'*** Reducing Thoriums ***'
			for j in range(len( filedata['REDUCEAS'] )):
				if filedata['REDUCEAS'][j] =='ARC':
					input_file=filedata['FILE'][j]
					print 'Processing %s'%input_file
					try:
						sp,unc=extract_arc(input_file=input_file)
						wave=arc_wavelength_solution(hdu)
						save_reduced_data(input_file,sp,unc,wave)
					except: print 'Extraction FAILED for %s' %input_file
			print'*** Reducing Iodine Flats ***'
			for j in range(len( filedata['REDUCEAS'] )):
				if filedata['REDUCEAS'][j] =='I2FLAT':
					input_file=filedata['FILE'][j]
					print 'Processing %s'%input_file
					try:
						sp,unc=extract_science_or_I2flat(input_file=input_file)
						save_reduced_data(input_file,sp,unc)
					except: print 'Extraction FAILED for %s' %input_file
			print'*** Reducing Stellar Exposures ***'
			for j in range(len( filedata['REDUCEAS'] )):
				if filedata['REDUCEAS'][j] =='SCIENCE':
					input_file=filedata['FILE'][j]
					print 'Processing %s'%input_file
					try:
						sp,unc=extract_science_or_I2flat(input_file=input_file)
						save_reduced_data(input_file,sp,unc)
					except: print 'Extraction FAILED for %s' %input_file	
			print'*** Reduction in %s completed ***'%child
			os.chdir(initial_path)
			

