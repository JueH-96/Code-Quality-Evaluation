# YOUR CODE HERE
S = input()
T = input()

positions = []
i = 0  # index in S
j = 0  # index in T

while i < len(S) and j < len(T):
    if S[i] == T[j]:
        positions.append(j + 1)  # 1-indexed position
        i += 1
    j += 1

print(' '.join(map(str, positions)))