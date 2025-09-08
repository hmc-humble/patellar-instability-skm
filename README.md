# temp-patellar-instability-skm
TODO temporary repo until manuscript is published

This repository contains two models:
1. a segmentation model to segment the femur, patella, and tibia from sagittal PD-weighted MRIs
2. a statistical knee model to quantify the segmented 3D bony anatomy

## Segmentation model
The segmentation model takes in a sagittal PD-weighted MRI DICOM series and outputs an NRRD segmentation.

Model weights must be downloaded from [Drive](https://drive.google.com/file/d/1Edn2ZWCvjQttj8sAsG302HOmrl17ySfB/view) and should be saved to `segmentation_model/weights/`.

Usage:

    sys.path.append('../segmentation_model')
    from inference import segment_image

    SERIES_DIR = <your_series_directory>
    OUTPUT_FILENAME = <your_desired_output_filename>
    segment_image(SERIES_DIR, OUTPUT_FILENAME)

Raw model outputs (such as the sample `examples/sample_data/segmentation/segmentation.nrrd`) should be manually reviewed and corrected using a software of your choice (e.g., [3D Slicer](https://www.slicer.org/)). We provide `examples/sample_data/segmentation/segmentation_corrected.nrrd` as an example of a manually corrected segmentation model output.

## Statistical knee model
With a knee segmentation in hand, we can now move on to quantifying the segmented 3D bony anatomy. The statistical knee model was constructed using 272 knees with patellar instability and 26 knees with ACL injuries. It quantifies bone positions (e.g., lateral displacement of the patella) relative to the femur and bone shape features. Bone shape features were learned through principal component analysis.

Usage:

    sys.path.append('../statistical_knee_model')
    from fit_to_new_knee import KneeReconstruction

    SEG_FILENAME = '../examples/sample_data/segmentation/segmentation_corrected.nrrd'
    RIGHT_KNEE = False

    recon = KneeReconstruction(SEG_FILENAME, right_knee=RIGHT_KNEE, verbose=True)
    recon.fit()
    recon.save()

This fit is produced through a series of steps:
    - the bone segmentations are converted into surface meshes (`examples/sample_data/mesh/<bone>.vtk`)
    - the surface meshes are rigidly registered to the mean model meshes such that the individual bones align
    - the statistical knee model parameters (bone positions and bone shape features) are optimized to minimize the chamfer distance between the knee's meshes and the mean model meshes

Three reconstructed knees are saved in `examples/sample_data/recon/`:
1. `<bone>_recon_registered_to_mean.vtk` - the reconstruction of the bone rigidly registered to corresponding mean bone
2. `<bone>_recon_registered_to_mean_femur.vtk` - the reconstruction of the bone such that it maintains its patient-specific relative alignment to the femur when the femur is rigidly registered to the mean femur
3. `<bone>_recon.vtk` - the reconstruction of the bone in its original orientation

Visualizations of these outputs are available in `visualizations/recons/`.

## Example
Example usage is provided in `examples/example.ipynb`.

## Feature visualizations
Statistical knee model feature morphs can be visualized in `visualizations/feature_morphs/`.

## Citation
If you use this model in your work, please cite the accompanying publication:

TODO
