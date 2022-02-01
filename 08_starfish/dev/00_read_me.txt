-----------------------------------------------------------------------------------------------------

Bienvenue pour ce dernier projet OpenClassrooms

Nous avons décidé de participer au concours Kaggle suivant:

https://www.kaggle.com/c/tensorflow-great-barrier-reef/data


** DESCRIPTION **
L'objectif de ce concours, est de prédire la présence et la position des étoiles de mer. Les prédictions doivent être des contours avec un label et un indice de confiance. Une image peut contenir aucun ou plusieurs étoiles de mer.



** FICHIERS **
train/ - Folder containing training set photos of the form video_{video_id}/{video_frame_number}.jpg.

[train/test].csv - Metadata for the images. As with other test files, most of the test metadata data is only available to your notebook upon submission. Just the first few rows available for download.

	video_id - ID number of the video the image was part of. The video ids are not meaningfully ordered.
	
	video_frame - The frame number of the image within the video. Expect to see occasional gaps in the frame number from when the diver surfaced.

	sequence - ID of a gap-free subset of a given video. The sequence ids are not meaningfully ordered.

	sequence_frame - The frame number within a given sequence.

	image_id - ID code for the image, in the format '{video_id}-{video_frame}'

	annotations - The bounding boxes of any starfish detections in a string format that can be evaluated directly with Python. Does not use the same format as the predictions you will submit. Not available in test.csv. A bounding box is described by the pixel coordinate (x_min, y_min) of its upper left corner within the image together with its width and height in pixels.

example_sample_submission.csv - A sample submission file in the correct format. The actual sample submission will be provided by the API; this is only provided to illustrate how to properly format predictions. The submission format is further described on the Evaluation page.

example_test.npy - Sample data that will be served by the example API.

greatbarrierreef - The image delivery API that will serve the test set pixel arrays. You may need Python 3.7 and a Linux environment to run the example offline without errors.



