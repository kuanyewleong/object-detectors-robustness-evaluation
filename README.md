# Evaluating the Robustness of Pretrained Object Detectors under Controlled Object Occlusion

## Project Overview

This project investigates how well pretrained object detectors can recognise everyday objects when they are partly hidden. We prepare a controlled test set using common items such as bottles, cups, books and potted plant, photographed from different viewpoints and backgrounds, then covered at several occlusion levels and positions.

Using fixed pretrained models with no retraining, we record detection results and confidence scores, calculate performance at each occlusion level, and identify common failure patterns. The study focuses on how object visibility, occlusion position, object category and detector architecture affect robustness.

## Research Focus

The central research question is:

> **How much occlusion can a pretrained object detector tolerate before its detection performance begins to fail?**

The project examines:

- Detection performance at increasing levels of occlusion.
- The effect of different occlusion positions, such as top, bottom, left, right and centre.
- Differences in robustness across everyday object classes.
- Differences in failure behaviour across detector architectures.
- Changes in detection confidence and localisation quality as objects become increasingly hidden.

## Pretrained Detectors

Three pretrained object detectors are used:

### 1. RF-DETR Nano
A compact RF-DETR variant representing a modern transformer-based object detector.

### 2. TorchVision MobileNetV3-SSDLite
A lightweight SSD-style detector using MobileNetV3 as its backbone, representing efficient CNN-based object detection.

### 3. YOLO26n
The Nano variant of YOLO26, representing a lightweight one-stage YOLO detector.

## Why These Models?

The selected detector classes and model variants were chosen with **edge devices and resource-constrained environments** in mind.

Rather than comparing the largest or most computationally expensive object detectors, this project focuses on compact models that are more realistic for deployment on hardware with limited compute, memory and power.

The three detectors also represent different object-detection design approaches:

| Detector | General Approach | Selected Variant |
|---|---|---|
| RF-DETR | Transformer-based detection | Nano |
| MobileNetV3-SSDLite | Lightweight CNN + SSD | MobileNetV3-SSDLite |
| YOLO26 | One-stage YOLO detection | Nano |

This allows the project to investigate whether different lightweight detector architectures respond differently to object occlusion while remaining relevant to practical deployment scenarios such as embedded systems, mobile platforms, robotics and other edge-AI applications.

## Experimental Approach

All three detectors evaluate the **same shared test set**.

The models remain fixed throughout the study:

- No retraining.
- No fine-tuning.
- No modification of pretrained weights.
- The same test images and ground-truth annotations are used for all detectors.
- Detector outputs are converted into a common evaluation format for comparison.

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

## Test Set

The test set uses common everyday object classes that can be recognised by all three pretrained detectors, such as:

- Bottle
- Cup
- Book
- Potted Plants

Objects are photographed under controlled variations including:

- Different physical object instances.
- Different viewpoints.
- Different backgrounds.
- Different occlusion percentages.
- Different occlusion positions.

Example occlusion levels may include:

- 0% — fully visible
- 20%
- 40%
- 60%
- 80%

Example occlusion positions may include:

- Top
- Bottom
- Left
- Right
- Centre

## Evaluation

The project may evaluate:

- Detection rate.
- Detection retention relative to the unobstructed baseline.
- Confidence score changes.
- Intersection over Union (IoU).
- Object-class-specific performance.
- Occlusion-position-specific performance.
- Failure cases such as missed detections, incorrect classifications and poor localisation.

A common evaluation pipeline is used so that differences in results reflect the behaviour of the detectors rather than differences in evaluation procedure.

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
