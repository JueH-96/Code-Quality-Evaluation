N = int(input())
A = list(map(int, input().split()))

ops = 0
while sum(1 for a in A if a > 0) > 1:
    A.sort(reverse=True)
    if A[0] > 0 and A[1] > 0:
        A[0] -= 1
        A[1] -= 1
        ops += 1
print(ops)