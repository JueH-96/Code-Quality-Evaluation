import sys
from itertools import permutations

input = sys.stdin.read
data = input().split()

N = int(data[0])
actions = [(int(data[i*2+1]), data[i*2+2]) for i in range(N)]

# Calculate the minimum fatigue level
def calculate_fatigue(sequence):
    left_hand = 0
    right_hand = 0
    fatigue = 0

    for key, hand in sequence:
        if hand == 'L':
            fatigue += abs(key - left_hand)
            left_hand = key
        else:
            fatigue += abs(key - right_hand)
            right_hand = key

    return fatigue

# Generate all possible sequences of actions
min_fatigue = float('inf')
for seq in permutations(actions):
    fatigue = calculate_fatigue(seq)
    if fatigue < min_fatigue:
        min_fatigue = fatigue

print(min_fatigue)