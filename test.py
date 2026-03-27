import pyautogui
import time

def auto_clicker(interval=60):
    print(f"Auto-clicker started. Clicking every {interval} seconds.")
    print("Move your mouse to the corner of the screen to emergency stop.")
    
    try:
        while True:
            # Get current mouse position
            x, y = pyautogui.position()
            # Perform a left-click at current location
            pyautogui.click(x, y)
            print(f"Clicked at ({x}, {y}) at {time.strftime('%H:%M:%S')}")
            # Wait for the specified interval
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nScript stopped manually.")

# Starts clicking every 60 seconds
auto_clicker(15)
