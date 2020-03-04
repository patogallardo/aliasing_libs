import envVars
import glob
import os

def getAllDataInDataframe(df):
    assert 'mce' in df # check that mce exists in df
    for filename, mce in zip(df.fname, df.mce):
        getData(filename, mce=mce)


def getData(fname, mce, debug=True):
    '''Checks if filename exists in destination directory.
    If it doesn't exist it will copy it to the local drive.'''
    copyTo = os.path.join(envVars.local_dataOrigin, mce)
    if debug:
        print "locating file %s" %fname
    if not os.path.exists(os.path.join(copyTo, fname) ):
        print "%s is not in the local folder" %fname
        fullPath = glob.glob(envVars.smb_dataOrigin + "/%s/*/%s" %(mce, fname)) #mce1/20190614/123123415_aliasing...
        if debug:
            print fullPath
        assert len(fullPath) == 1 # check duplicates or if samba drive is mounted
        fullPath = fullPath[0]
        print "found fname in the samba drive"
        command = 'rsync -ravz --progress %s %s' %(fullPath, copyTo)
        print "Now transfering file..."
        print command
        os.system(command)

    if not os.path.exists(os.path.join(copyTo, fname+'.run')):
        #now look for run file...
        fullPath = glob.glob(envVars.smb_dataOrigin + "/%s/*/%s.run" %(mce, fname))
	if len(fullPath) > 1:
            fullPath = fullPath[0]
            print "found runfile in the samba drive"
            command = 'rsync -ravz --progress %s %s' %(fullPath, copyTo)
            print "Now transfering file..."
            print command
            os.system(command)
            print "Done transfering"
