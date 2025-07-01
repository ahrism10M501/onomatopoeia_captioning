from ultralytics import YOLO

root_path = ''

def main():
    model = YOLO("yolov8m.yaml") 

    model.train(
        data=f"{root_path}\\_yolo.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        workers=8,  
        device=0 
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # 윈도우 실행 시 필수
    main()
