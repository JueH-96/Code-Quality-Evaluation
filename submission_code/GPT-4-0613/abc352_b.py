# YOUR CODE HERE
S = input().strip()
T = input().strip()

i = 0
j = 0
positions = []

while i < len(S) and j < len(T):
    if S[i] == T[j]:
        positions.append(j+1)
        i += 1
    j += 1

print(' '.join(map(str, positions)))