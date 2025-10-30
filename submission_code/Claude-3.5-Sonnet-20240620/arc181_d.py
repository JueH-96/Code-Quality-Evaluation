# YOUR CODE HERE
def count_inversions(arr):
    return sum(arr[i] > arr[j] for i in range(len(arr)) for j in range(i + 1, len(arr)))

def apply_operation(arr, k):
    for i in range(k - 1):
        if arr[i] > arr[i + 1]:
            arr[i], arr[i + 1] = arr[i + 1], arr[i]

N = int(input())
P = list(map(int, input().split()))
M = int(input())
A = list(map(int, input().split()))

for operation in A:
    apply_operation(P, operation)
    print(count_inversions(P))