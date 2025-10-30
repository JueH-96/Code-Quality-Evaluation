# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Calculate the number of distinct values in the subsequence (A_l, A_{l+1}, \ldots, A_r)
def f(l, r):
    return len(set(A[l:r+1]))

# Evaluate the expression
result = 0
for i in range(N):
    for j in range(i, N):
        result += f(i, j)

# Print the answer
print(result)