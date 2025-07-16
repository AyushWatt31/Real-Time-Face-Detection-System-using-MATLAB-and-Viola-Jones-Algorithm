% Initialize face detector and webcam
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
faceDetector.MergeThreshold = 5;  % Adjust for accuracy
cam = webcam;  % Use default webcam

% Create figure window
figure;
set(gcf, 'Name', 'Real-Time Face Detection', 'NumberTitle', 'off');

while true
    % Capture frame from webcam
    frame = snapshot(cam);
    grayFrame = rgb2gray(frame);
    grayFrame = imadjust(grayFrame);  % Enhance contrast

    % Detect all faces
    bboxes = step(faceDetector, grayFrame);

    % If faces are detected
    if ~isempty(bboxes)
        % Pick the largest one
        areas = bboxes(:,3) .* bboxes(:,4);
        [~, idx] = max(areas);
        bestBox = bboxes(idx, :);

        % Annotate
        result = insertObjectAnnotation(frame, 'rectangle', bestBox, 'Face');

        % Display selected bounding box only (in Command Window)
        clc;  % Clear previous outputs
        disp('Detected Face Bounding Box:');
        disp(bestBox);
    else
        result = frame;  % Show frame without annotation
        clc;
        disp('No face detected');
    end

    % Show the frame
    imshow(result);
    drawnow;
end
