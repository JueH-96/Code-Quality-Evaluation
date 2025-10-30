# YOUR CODE HERE

# Read the inputs
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort the sequences
A.sort()
B.sort()

# Check if the sequences contain two consecutive elements from A
for i in range(N - 1):
    if A[i] + 1 == A[i + 1]:
        print("Yes")
        break
else:
    print("No")