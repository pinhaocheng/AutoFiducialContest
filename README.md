# AutoFiducialContest
The repository contains the example data and code stubs for the Spring 2025 coding contest hosted by Openwater.

## Objective
The objective of this contest is to develop an algorithm or software solution capable of automatically identifying and extracting the 3D coordinates of specific fiducial points (landmarks) from a given 3D facial mesh acquired with photogrammetry. These landmarks correspond to key anatomical locations on the human face used in registration.

## Background
Photogrammetry can be used to extract shape-accurate reconstruction of three dimensional objects. Openwater is researching the use of photogrammetry as a possible solution for determining the positioning of therapeutic ultrasound transducers on subjects’ heads. 

One key requirement is the ability to register the photogrammetry scan into the subjects’ MRI space, as photogrammetry reconstructions can have arbitrary scaling and orientation. Our current proposed method is to use an initial fiducial registration of facial landmarks, followed by a precise surface-to-surface Iterative Closest Point (ICP) registration.  

For the initial fiducial registration to work, facial landmarks must be identified on the photogrammetry scans and in the subject’s MRI. For the photogrammetry scans acquired during operation, this can be a time-consuming process for a system operator if the points must be placed “by hand”, and it would be beneficial to have an algorithm that can automatically place these landmarks so that the operator must only adjust and approve them to continue to fiducial registration.

## Data Provided
Example 3D meshes (.obj)
Corresponding 3D coordinates of fiducial points (LPS)

## Evaluation
- Contestants submit Python code.
- Code evaluated on an unseen dataset of 3D head scans.
- Accuracy will be assesed by mean-squared-error of positional offset between predicted and manually-marked points.

## Resources Provided
- Photogrammetry scanning application
- Reconstruction pipeline template

## Technical Requirements
- Solution must be written in Python. The top level script is `find_fiducials.py`, which executes the corresponding `find_fiducials` function. Your solution should be accessible via the `find_fiducials` function.
- Additional input parameters may be added to tune the algorithm, but the default values will be considered your submission.
- Solution must run locally on a PC.
- Can utilize external binaries or GPU acceleration, but installation instructions must be clear and executables must be free of malware.
  
### Fiducial Points
Fiducial points are defined in the coordinate system of the mesh. They are:
- Left ear (the tip of the tragus)
- Right ear (the tip of the tragus)
- Inside edge of left eye
- Outside edge of left eye
- Inside edge of right eye
- Outside edge of right eye
- Nasion (most concave point on the bridge of the nose between the eyes)

### Potential Challenges
Variations in facial geometry: Human faces exhibit a wide range of variations in terms of shape, size, and features. The algorithm should be robust to these variations.
Noise and artifacts: The 3D mesh may contain noise or artifacts due to the scanning process. The algorithm should be able to filter out noise and extract landmarks accurately.
Computational efficiency: The algorithm should be computationally efficient to enable real-time or near-real-time processing of 3D facial data.

### Potential Approaches
Machine learning-based methods: Deep learning models can be trained on large datasets of 3D facial meshes to learn to identify and extract landmarks accurately.
Model-based methods: 3D face models can be fitted to the input mesh, and landmarks can be extracted based on the model parameters.
Hybrid methods: Combining machine learning and model-based approaches may leverage the strengths of both.

## Acknowledgement of Open Source 
By participating, you agree that your submission will be licensed under the AGPLv3 license. You certify that your work complies with AGPL requirements, including providing source code for any modifications. You retain copyright but grant Openwater the right to distribute your submission under AGPL terms.  By submitting code under AGPLv3, participants irrevocably waive any right to charge Openwater or its licensees fees, royalties, or compensation for the use, distribution, or modification of their submission. This includes prohibiting claims related to patent licensing, copyright monetization, or other intellectual property assertions arising from AGPLv3 compliance.  

## $5000 Winner Determination
Submissions will be evaluated on a dataset of 3D head scans that are not provided to contestants during the contest. For each scan, the submitted code will predict the 3D positions of the defined fiducial points. These predicted positions will be compared to the ground truth positions that were manually marked. The error metric for evaluation will be the mean squared error (MSE), calculated based on the total positional offset between the predicted and ground truth fiducial points across all landmarks and all scans in the evaluation dataset. The submission with the lowest MSE will be declared the winner.
