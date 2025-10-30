# Read the input
N, S = map(int, input().split())
A = list(map(int, input().split()))

# Check if there exists a contiguous subsequence with sum S
total = 0
for i in range(N):
    total += A[i]
    if total == S:
        print("Yes")
        exit()
    if total > S:
        total -= A[i - N + 1]

# If no such subsequence is found, print "No"
print("No")