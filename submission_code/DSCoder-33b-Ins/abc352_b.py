# YOUR CODE HERE

S = input()
T = input()

j = 0
correct_positions = []

for i in range(len(T)):
    if j < len(S) and T[i] == S[j]:
        correct_positions.append(i + 1)
        j += 1

print(*correct_positions)