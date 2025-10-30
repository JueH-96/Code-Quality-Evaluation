# Read the input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Create a set of unique elements in A and B
unique_elements = set(A + B)

# Check if the length of the unique elements is equal to N
if len(unique_elements) != N:
    print("No")
    exit()

# Create a sequence X of length N consisting of 0 and 1
X = [0] * N
for i in range(N):
    if i+1 in A and i+1 in B:
        print("No")
        exit()
    elif i+1 in A:
        X[i] = 0
    else:
        X[i] = 1

# Check if (A, B) is a good pair of sequences
for i in range(M):
    if X[A[i]-1] == X[B[i]-1]:
        print("No")
        exit()

print("Yes")