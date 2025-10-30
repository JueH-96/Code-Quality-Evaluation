# YOUR CODE HERE
N = int(input())
positions = [0] * (2 * N + 1)  # Using 1-based indexing

for idx in range(1, N + 1):
    A_i, B_i = map(int, input().split())
    if A_i > B_i:
        A_i, B_i = B_i, A_i
    positions[A_i] = idx
    positions[B_i] = idx

stack = []
for i in range(1, 2 * N + 1):
    chord = positions[i]
    if not stack or stack[-1] != chord:
        stack.append(chord)
    else:
        stack.pop()

if stack:
    print("Yes")
else:
    print("No")