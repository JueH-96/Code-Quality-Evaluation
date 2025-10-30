# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().strip().split()
S = data[0]
T = data[1]

correct_indices = []
s_index = 0

for t_index, char in enumerate(T):
    if s_index < len(S) and char == S[s_index]:
        correct_indices.append(t_index + 1)
        s_index += 1

print(" ".join(map(str, correct_indices)))