# Note to guide the students for this project

The students' main responsibilities therefore centre on:

1. Designing and preparing the controlled test set.
2. Collecting and organising images.
3. Annotating ground-truth objects.
4. Generating controlled occlusion conditions.
5. Running inference using the assigned pretrained detector.
6. Recording predictions, confidence scores and bounding boxes.
7. Evaluating detector performance.
8. Analysing failure cases.
9. Comparing results across the three detectors.
10. Presenting findings through graphs, tables, a research report and a final presentation.

## Important note for photoshooting
One important experimental detail: because the students will photograph cups, books, and potted plants etc., we recommend keeping the camera and target object fixed while adding/removing the occluder. Then the bounding box from the unobstructed image can be reused as the ground-truth full-object box for its occluded versions. This willbe the baseline we needed for benchmarking.

## Team Workflow

The project is designed for three students working collaboratively on one shared research problem.

- **Student A:** RF-DETR Nano
- **Student B:** TorchVision MobileNetV3-SSDLite
- **Student C:** YOLO26n

All students jointly contribute to test-set preparation, annotation, evaluation design, comparative analysis and final presentation.


## Expected Deliverables

By the end of the project, the team is expected to produce:

- A structured controlled-occlusion image test set.
- Ground-truth annotations and metadata.
- Detection results from all three pretrained models.
- Comparative evaluation tables.
- Graphs showing performance under increasing occlusion.
- A visual gallery of successful and failed detections.
- Cross-detector robustness analysis.
- A short research report.
- A research poster.
- A group presentation.

## Note

This is part of the **Malaysia Science Scholar’s Programme (MYSSP)**, Malaysia’s first science research programme for pre-university students. MYSSP, crafted by **MYResearchGuide**, is an 8-week free mentorship programme pairing students with experienced researchers under a selection of STEM-based projects.