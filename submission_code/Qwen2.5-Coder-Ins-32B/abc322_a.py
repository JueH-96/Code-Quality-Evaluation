# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
S = input[1]

def find_abc_position(S):
    for i in range(len(S) - 2):
        if S[i:i+3] == "ABC":
            return i + 1
    return -1

print(find_abc_position(S))