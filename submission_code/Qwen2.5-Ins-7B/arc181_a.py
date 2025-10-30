# YOUR CODE HERE
def min_operations(n, perm):
    operations = 0
    i = 0
    while i < n:
        if perm[i] == i + 1:
            i += 1
        else:
            j = i
            while j < n and perm[j] != i + 1:
                j += 1
            if j == n:
                return -1
            operations += 1
            if i + 1 < j:
                perm[i:j+1] = sorted(perm[i:j+1])
            i = j + 1
    return operations

T = int(input())
for _ in range(T):
    N = int(input())
    perm = list(map(int, input().split()))
    print(min_operations(N, perm))