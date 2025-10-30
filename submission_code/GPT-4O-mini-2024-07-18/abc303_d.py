def min_time_to_type_string(X, Y, Z, S):
    total_time = 0
    caps_lock_on = False

    for char in S:
        if char == 'A':
            if caps_lock_on:
                total_time += X  # Press 'a' key (since Caps Lock is on, it types 'A')
            else:
                total_time += Y  # Press 'Shift + a' key to type 'A'
        elif char == 'a':
            if caps_lock_on:
                total_time += Y  # Press 'Shift + a' key to type 'a' (since Caps Lock is on)
            else:
                total_time += X  # Press 'a' key to type 'a'
        
        # After processing the character, decide if we need to toggle Caps Lock
        if char == 'A' and not caps_lock_on:
            # If we just typed 'A' and Caps Lock was off, we might want to turn it on
            if Z < Y:  # Check if toggling is cheaper than pressing Shift
                total_time += Z  # Toggle Caps Lock on
                caps_lock_on = True
        elif char == 'a' and caps_lock_on:
            # If we just typed 'a' and Caps Lock was on, we might want to turn it off
            if Z < X:  # Check if toggling is cheaper than pressing 'a'
                total_time += Z  # Toggle Caps Lock off
                caps_lock_on = False

    return total_time

import sys
input = sys.stdin.read
data = input().strip().splitlines()
X, Y, Z = map(int, data[0].split())
S = data[1]

result = min_time_to_type_string(X, Y, Z, S)
print(result)