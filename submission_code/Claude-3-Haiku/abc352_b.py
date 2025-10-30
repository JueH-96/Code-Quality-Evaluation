import sys

S = input()
T = input()

correct_positions = []
j = 0
for i in range(len(S)):
    if j < len(T) and S[i] == T[j]:
        correct_positions.append(j+1)
        j += 1
    else:
        j += 1

print(" ".join(map(str, correct_positions)))