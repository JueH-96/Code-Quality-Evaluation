# YOUR CODE HERE
import sys
data = sys.stdin.read().split()
S = data[0]
T = data[1]
len_S = len(S)
positions = []
j = 0
for i in range(len_S):
    while j < len(T) and T[j] != S[i]:
        j += 1
    positions.append(j)
    j += 1
output_str = ' '.join(str(pos + 1) for pos in positions)
print(output_str)