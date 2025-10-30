# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
keys = [list(map(str, sys.stdin.readline().strip().split())) for _ in range(N)]

left_key = 1
right_key = 100
fatigue = 0

for key, hand in keys:
    key = int(key)
    if hand == 'L':
        if key != left_key:
            fatigue += abs(key - left_key)
            left_key = key
    else:
        if key != right_key:
            fatigue += abs(key - right_key)
            right_key = key

print(fatigue)