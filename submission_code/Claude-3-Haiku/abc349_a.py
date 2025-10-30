# Read the input
N = int(input())
A = [int(x) for x in input().split()]

# Calculate the final score of person N
score_n = 0
for i in range(N-1):
    score_n += N - i - 1 - 2 * A[i]

print(score_n)