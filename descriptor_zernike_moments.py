# import the necessary packages
import mahotas as mh
 
class ZernikeMoments:
	def __init__(self, radius, degree):
		# Store the size of the radius that will be
		# used when computing moments
		self.radius = radius
		self.degree = degree
 
	def describe(self, image):
		# Return the Zernike moments for the image
		# http://mahotas.readthedocs.io/en/latest/api.html#mahotas.features.zernike_moments
		# mahotas.features.zernike_moments(im, radius, degree=8, cm={center_of_mass(im)})
		#x, y = image.shape[1]//2, image.shape[0]//2
		# return mh.features.zernike_moments(image, self.radius, cm=(y, x))
		return mh.features.zernike_moments(image, radius=self.radius, degree=self.degree)