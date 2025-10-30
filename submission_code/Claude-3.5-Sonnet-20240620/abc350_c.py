# YOUR CODE HERE
def swap(arr, i, j):
    arr[i-1], arr[j-1] = arr[j-1], arr[i-1]

N = int(input())
A = list(map(int, input().split()))

operations = []
for i in range(1, N+1):
    if A[i-1] != i:
        j = A.index(i) + 1
        swap(A, i, j)
        operations.append((i, j))

print(len(operations))
for op in operations:
    print(f"{op[0]} {op[1]}")