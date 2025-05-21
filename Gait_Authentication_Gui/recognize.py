def recognize_from_camera():
    cap = cv2.VideoCapture(0)  # Webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame temporarily
        cv2.imwrite("temp_frame.jpg", frame)
        # Recognize gait
        result = recognize("temp_frame.jpg")
        cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Gait Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()