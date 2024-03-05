import numpy as np
import cv2


def makeAPointPair(x1, x2):
	u = x1[0]
	v = x1[1]
	x = x2[0]
	y = x2[1]
	A = np.array([[
			x, y, 1, 0, 0, 0, -x*u, -y*u, -u
		],[
			0, 0, 0, x, y, 1, -x*v, -y*v, -v
		]]
	)
	return A

def makeA(x1, x2):
	A = []
	for i in range(x1.shape[0]):
		A.append(makeAPointPair(x1[i,:], x2[i,:]))
	return np.vstack(A)

def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	A = makeA(x1, x2)
	U, S, Vh = np.linalg.svd(A)
	H2to1 = np.reshape(Vh.T[:, 8], (3,3))
	return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
	x1_centroid = np.mean(x1, axis = 0)
	x2_centroid = np.mean(x2, axis = 0)

	#Shift the origin of the points to the centroid
	x1 = x1-x1_centroid
	x2 = x2-x2_centroid

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	scale1 = np.sqrt(2)/np.max(np.linalg.norm(x1, axis = 1))
	scale2 = np.sqrt(2)/np.max(np.linalg.norm(x2, axis = 1))
	x1 = scale1*x1
	x2 = scale2*x2
	#Similarity transform 1
	T1 = np.array([[scale1, 0, -x1_centroid[0]*scale1],
				[0, scale1, -x1_centroid[1]*scale1],
				[0, 0, 1]])

	#Similarity transform 2
	T2 = np.array([[scale2, 0, -x2_centroid[0]*scale2],
				[0, scale2, -x2_centroid[1]*scale2],
				[0, 0, 1]])

	#Compute homography
	H = computeH(x1, x2)

	#Denormalization
	H2to1 = np.linalg.inv(T1)@H@T2

	return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

	np.random.RandomState(1337)

	bestH2to1 = np.zeros((3,3))
	inliers = np.zeros(len(locs1))
	best_score = 0
	for i in range(max_iters):
		# pick 4 points in each image
		rand_4_idx = np.random.choice(range(len(locs1)), 4, replace = False)
		x1 = locs1[rand_4_idx, :]
		x2 = locs2[rand_4_idx, :]
		# compute H using computeH norm with those 4 pairs of points
		try:
			H = computeH_norm(x1, x2) # In a try block incase one of the coordinates is all 0 and divide by 0 starts happening.
		except:
			continue

		# Use the computed H and warp x2 to x1 (image 2 to image 1).
		l2 = np.vstack((locs2.T, np.ones((locs2.shape[0]))))
		locs2_pred = H@l2
		locs2_pred = locs2_pred[:2, :]/locs2_pred[2,:]

		dist_matrix = np.linalg.norm(locs1.T-locs2_pred, axis = 0)

		# compute inlier using all matched points
		inliers_temp = np.where(dist_matrix < inlier_tol, 1, 0)
		score = np.sum(inliers_temp)

		if score > best_score:
			inliers = inliers_temp
			bestH2to1 = H
			best_score = score

	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	# x_template = H2to1*x_photo

	#For warping the template to the image, we need to invert it.
	# H_inv = np.linalg.pinv(H2to1) # Not needed for this function?? Why are we asked to do this.....

	#Create mask of same size as template
	mask = np.ones(template.shape)

	#Warp mask by appropriate homography
	warped_mask = cv2.warpPerspective(mask, H2to1, [img.shape[1], img.shape[0]])

	#Warp template by appropriate homography
	warped_template = cv2.warpPerspective(template, H2to1, [img.shape[1], img.shape[0]])

	#Use mask to combine the warped template and the image
	composite_img = img
	composite_img[warped_mask != 0] = warped_template[warped_mask != 0]
	
	return composite_img


