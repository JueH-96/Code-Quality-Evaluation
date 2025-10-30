from bisect import bisect_left, bisect_right

N = int(input())
A = list(map(int, input().split()))

# Preprocess the array to find the next and previous greater elements
next_greater = [N] * N
prev_greater = [-1] * N
stack = []

for i in range(N):
    while stack and A[stack[-1]] < A[i]:
        next_greater[stack.pop()] = i
    if stack:
        prev_greater[i] = stack[-1]
    stack.append(i)

# Calculate the contribution of each element
contribution = [0] * N
for i in range(N):
    right = next_greater[i]
    left = prev_greater[i]
    if right != N:
        contribution[right] += (i - left) * A[i]
    if left != -1:
        contribution[i] -= (right - i) * A[left]

# Calculate the answer using the contribution array
answer = sum(max(0, c) for c in contribution)
print(answer)