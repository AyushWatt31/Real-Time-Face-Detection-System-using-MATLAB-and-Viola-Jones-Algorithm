faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
faceDetector.MergeThreshold = 5;
cam = webcam;

figure;
set(gcf, 'Name', 'Real-Time Face Detection', 'NumberTitle', 'off');

while true
    frame = snapshot(cam);
    grayFrame = rgb2gray(frame);
    grayFrame = imadjust(grayFrame);
    bboxes = step(faceDetector, grayFrame);
    
    if ~isempty(bboxes)
        areas = bboxes(:,3) .* bboxes(:,4);
        [~, idx] = max(areas);
        bestBox = bboxes(idx, :);

        result = insertObjectAnnotation(frame, 'rectangle', bestBox, 'Face');

        clc;
        disp('Detected Face Bounding Box:');
        disp(bestBox);
    else
        result = frame;
        clc;
        disp('No face detected');
    end

    imshow(result);
    drawnow;
end
