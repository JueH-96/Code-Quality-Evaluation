import sys
input = sys.stdin.read

def solve():
    data = input().split()
    X = int(data[0])
    Y = int(data[1])
    Z = int(data[2])
    S = data[3]
    
    # Initial state
    caps_lock_on = False
    total_time = 0
    
    for char in S:
        if char == 'a':
            if caps_lock_on:
                # Caps lock is on, need to use Shift+'a' to get 'a'
                total_time += Y
            else:
                # Caps lock is off, just press 'a'
                total_time += X
        elif char == 'A':
            if caps_lock_on:
                # Caps lock is on, just press 'a'
                total_time += X
            else:
                # Caps lock is off, need to use Shift+'a' to get 'A'
                total_time += Y
    
    # Calculate the cost of using Caps Lock toggle
    # We need to determine if toggling Caps Lock could be cheaper in some cases
    # We will simulate the process with and without initial Caps Lock toggle
    min_time = total_time
    
    # Case 1: Start with Caps Lock off (already calculated)
    time_with_caps_off = total_time
    
    # Case 2: Start with Caps Lock on
    time_with_caps_on = Z  # Initial cost to turn on Caps Lock
    for char in S:
        if char == 'a':
            # Caps lock is on, need to use Shift+'a' to get 'a'
            time_with_caps_on += Y
        elif char == 'A':
            # Caps lock is on, just press 'a'
            time_with_caps_on += X
    
    # Compare both scenarios
    min_time = min(time_with_caps_off, time_with_caps_on)
    
    print(min_time)