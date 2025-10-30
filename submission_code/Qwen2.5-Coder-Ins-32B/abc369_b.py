# YOUR CODE HERE
def min_fatigue(N, actions):
    def calculate_fatigue(left_start, right_start):
        left_pos, right_pos = left_start, right_start
        fatigue = 0
        for key, hand in actions:
            if hand == 'L':
                fatigue += abs(key - left_pos)
                left_pos = key
            else:
                fatigue += abs(key - right_pos)
                right_pos = key
        return fatigue

    min_fatigue = float('inf')
    for left_start in range(1, 101):
        for right_start in range(1, 101):
            min_fatigue = min(min_fatigue, calculate_fatigue(left_start, right_start))
    
    return min_fatigue

import sys
input = sys.stdin.read().split()
N = int(input[0])
actions = [(int(input[2*i+1]), input[2*i+2]) for i in range(N)]

print(min_fatigue(N, actions))