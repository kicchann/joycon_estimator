import time
import threading
from src import JoyConEstimator

def main():
    estimator = JoyConEstimator()
    estimator.connect()
    thread = threading.Thread(target=estimator.start)
    thread.start()
    time.sleep(60*2)
    estimator.stop()

if __name__ == "__main__":
    main()
