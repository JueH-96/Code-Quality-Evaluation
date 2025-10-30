# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

MOD = 998244353

# Function to count arithmetic subsequences of length k
def count_arithmetic_subsequences(A, k):
    count = 0
    for i in range(N - k + 1):
        diff = A[i + 1] - A[i]
        for j in range(i + 2, N - k + 2):
            if A[j] - A[j - 1] == diff:
                count += 1
            else:
                break
    return count

# Calculate and print the number of arithmetic subsequences for each length k
for k in range(1, N + 1):
    print(count_arithmetic_subsequences(A, k), end=' ')