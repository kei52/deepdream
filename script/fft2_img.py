from import_list import *

print os.listdir('/home/roboworks/deepdream/image/')
for i in os.listdir('/home/roboworks/deepdream/image/'):
    if ('jpg' not in i) and ('png' not in i):
        print i

input_name = raw_input('>>')
directory = os.listdir('/home/roboworks/deepdream/image/{}'.format(input_name))

if os.path.exists('/home/roboworks/deepdream/image/{}'.format(input_name)+'/fft2') == False:
    os.mkdir('/home/roboworks/deepdream/image/{}'.format(input_name)+'/fft2')

print directory
for img_name in directory:
    if 'jpg' in img_name:
        img = cv2.imread('/home/roboworks/deepdream/image/{}/{}'.format(input_name,img_name),0)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        #CV2_change_PIL
        ccp1 = Image.fromarray(magnitude_spectrum)
        ccp1 = ccp1.convert('RGB')
        ccp1.save('/home/roboworks/deepdream/image/{}/fft2/{}_fft2.jpg'.format(input_name,img_name))

'''
plt.imshow(ccp1,cmap = 'gray')
plt.show()
'''

'''
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
'''
