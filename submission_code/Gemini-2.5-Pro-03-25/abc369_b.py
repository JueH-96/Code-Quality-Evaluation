# YOUR CODE HERE
import sys

def solve():
    # Read the number of key presses
    N = int(sys.stdin.readline())
    
    presses = []
    # Read all the key press information (key number and hand used)
    # Store them in a list of dictionaries for easy access
    for _ in range(N):
        line = sys.stdin.readline().split()
        presses.append({'key': int(line[0]), 'hand': line[1]})

    # Find the index of the first press for the left hand ('L') and the right hand ('R')
    # Initialize indices to -1 to indicate they haven't been found yet.
    first_L_idx = -1
    first_R_idx = -1

    # Iterate through the recorded presses to find the first occurrence for L and R
    for i in range(N):
        # If the current press uses the left hand and we haven't found the first L press yet
        if presses[i]['hand'] == 'L' and first_L_idx == -1:
            # Record the index of this first L press
            first_L_idx = i
        
        # If the current press uses the right hand and we haven't found the first R press yet
        if presses[i]['hand'] == 'R' and first_R_idx == -1:
            # Record the index of this first R press
            first_R_idx = i
        
        # Optimization: If we have found the first index for both hands, 
        # we can stop searching early. This is a minor optimization since N <= 100.
        if first_L_idx != -1 and first_R_idx != -1:
             break

    # Initialize the current positions of the left and right hands.
    # The optimal strategy is to start each hand on the first key it needs to press.
    # This minimizes the initial movement cost to 0.
    # We use a default position (e.g., key 1) if a hand is never used. This default value
    # is arbitrary and won't affect the fatigue calculation because the fatigue for a hand
    # only increases when it moves, which only happens if it's used.
    L_curr = 1 
    R_curr = 1 

    # If the left hand is used at least once (first_L_idx != -1),
    # set its initial position to the key it presses first.
    if first_L_idx != -1:
        L_curr = presses[first_L_idx]['key']
    
    # If the right hand is used at least once (first_R_idx != -1),
    # set its initial position to the key it presses first.
    if first_R_idx != -1:
        R_curr = presses[first_R_idx]['key']

    # Initialize the total fatigue level to 0.
    total_fatigue = 0

    # Simulate the performance step by step to calculate the total fatigue.
    # Iterate through each press in the sequence.
    for i in range(N):
        key_to_press = presses[i]['key']
        hand_to_use = presses[i]['hand']

        # If the left hand is used for this press
        if hand_to_use == 'L':
            # Calculate the fatigue increase from moving the left hand
            # from its current position (L_curr) to the target key (key_to_press).
            # The fatigue increase is the absolute difference in key numbers.
            fatigue_increase = abs(key_to_press - L_curr)
            # Add this increase to the total fatigue accumulated so far.
            total_fatigue += fatigue_increase
            # Update the left hand's current position to the key it just pressed.
            L_curr = key_to_press
        # If the right hand is used for this press
        else: # hand_to_use == 'R'
            # Calculate the fatigue increase for moving the right hand.
            fatigue_increase = abs(key_to_press - R_curr)
            # Add this increase to the total fatigue.
            total_fatigue += fatigue_increase
            # Update the right hand's current position.
            R_curr = key_to_press

    # After processing all N presses, print the final minimum total fatigue level.
    print(total_fatigue)

# Call the solve function to execute the logic based on standard input.
solve()