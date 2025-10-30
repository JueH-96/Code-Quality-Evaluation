# YOUR CODE HERE

# Read the inputs
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Initialize a list to store the count of each number in A and B
count = [0] * (N + 1)

# Count the occurrences of each number in A and B
for i in range(M):
    count[A[i]] += 1
    count[B[i]] += 1

# Check if there is a number that appears more than once in A and B
for i in range(1, N + 1):
    if count[i] > 1:
        print("No")
        break
else:
    print("Yes")