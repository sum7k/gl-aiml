import cv2 as cv

font = cv.FONT_HERSHEY_SIMPLEX

def draw_boxes(image, result):
	N = len(result['rois'])
	print("In N", N)
	if not N:
		return
	print("After N", N)
	for i in range(N):
		box = result['rois'][i]
		score = result['scores'][i]
		cv.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 3)
		cv.putText(image, '{0:.2f}'.format(score), (box[1], box[0] - 10), font, .7, (0,255,0), 3 ,cv.LINE_AA)
		print("drawn rectangles")
