import sys
from itertools import permutations

input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
functions = [(int(data[2*i+2]), int(data[2*i+3])) for i in range(N)]

def evaluate_sequence(sequence):
    value = 1
    for idx in sequence:
        A, B = functions[idx]
        value = A * value + B
    return value

max_value = float('-inf')
for perm in permutations(range(N), K):
    max_value = max(max_value, evaluate_sequence(perm))

print(max_value)