# computer-vision-fingertip-tracking
Implemented using OpenCV 4 and Python 3.7.

The algorithm has two parts:
- Hand segmentation
- Fingertip detection

## Hand segmentation
To achieve hand segmentation, a combination of background subtraction and back projection are used.
Background subtraction is achieved by finding the average of the first X number of frames before the user's hand enters the frame.
The average background frame is then subtracted from the input frame before thresholding to attain an accurate binary image (black and white) containing the user's hand and arm.
To remove the arm from the image and leave only the hand, back projection is used. A histogram of the user's hand is retrieved by prompting the user
to place their hand in a region of interest (ROI) in the frame. Using back projection and this histogram, the rough skin regions
within the frame are returned. This is only used to find the lowest point of the user's hand in the frame.
Every point in the binary image attained by background subtraction that is below this point is discarded, leaving just the hand.
The largest contour in the remaining image is returned, corresponding to the user's hand.

<img width="438" alt="Screenshot 2021-08-29 at 2 26 25 PM" src="https://user-images.githubusercontent.com/29169894/131236176-70ac9f4f-566d-4ddf-af21-2646bd277ae6.png">


## Fingertip detection
A convex hull around the hand contour is first found. Since fingertips are highly rounded, there is a cluster of hull points
around the fingertips. These are grouped so that each fingertip corresponds to a single hull point. The significant convexity defects are then found,
which identify the finger valleys. Letting X be the number of defects, the X + 1 topmost hull points can be identified as fingertips (assuming the hand is upright).
The locations of the fingertips are then returned.

<img width="419" alt="Screenshot 2021-08-29 at 2 26 52 PM" src="https://user-images.githubusercontent.com/29169894/131236178-a07c7b52-05b4-4491-878b-29388ff736c3.png">

