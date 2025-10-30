# YOUR CODE HERE
N = int(input())
A = [[]]  # Index from 1
for i in range(1, N+1):
    row = list(map(int, input().split()))
    A.append([0] + row)  # Add dummy at index 0

def combine(c, k):
    if c >= k:
        return A[c][k]
    else:
        return A[k][c]

current_element = 1
for k in range(1, N+1):
    current_element = combine(current_element, k)

print(current_element)