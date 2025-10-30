# Read the input
N = int(input())
A = list(map(int, input().split()))

# Check if there is a place where the same element appears three or more times in a row
for i in range(N-2):
    if A[i] == A[i+1] == A[i+2]:
        print("Yes")
        break
else:
    print("No")