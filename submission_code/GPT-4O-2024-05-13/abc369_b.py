# YOUR CODE HERE

def min_fatigue(N, actions):
    # Initialize positions of left and right hands
    left_pos = right_pos = None
    fatigue = 0
    
    for i in range(N):
        key, hand = actions[i]
        
        if hand == 'L':
            if left_pos is None:
                left_pos = key
            else:
                fatigue += abs(left_pos - key)
                left_pos = key
        else:
            if right_pos is None:
                right_pos = key
            else:
                fatigue += abs(right_pos - key)
                right_pos = key
    
    return fatigue

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
actions = [(int(data[i*2+1]), data[i*2+2]) for i in range(N)]

# Compute and print the result
print(min_fatigue(N, actions))