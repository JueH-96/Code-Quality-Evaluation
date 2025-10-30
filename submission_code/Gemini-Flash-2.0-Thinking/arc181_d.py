def count_inversions(arr):
    count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                count += 1
    return count

def apply_operation_k(p, k):
    for i in range(k - 1):
        if p[i] > p[i + 1]:
            p[i], p[i + 1] = p[i + 1], p[i]
    return p

n = int(input())
p = list(map(int, input().split()))
m = int(input())
a = list(map(int, input().split()))

current_p = list(p)

for k_op in a:
    apply_operation_k(current_p, k_op)
    inversions = count_inversions(current_p)
    print(inversions)