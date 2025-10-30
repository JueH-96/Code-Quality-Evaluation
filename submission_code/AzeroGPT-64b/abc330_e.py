import sys
from array import array  # noqa: F401

sys.setrecursionlimit(10 ** 7)
f_inf = float('inf')
mod = 10 ** 9 + 7

def resolve():
    n, q = map(int, input().split())
    A = list(map(int, input().split()))
    AB = [tuple(map(int, input().split())) for _ in range(q)]

    res = [0] * n
    A_set = set(A)
    for num in range(n):
        res[num] = num
        if num in A_set:
            res[num] = -1

    for i in range(n):
        if res[i] == -1:
            res[i] = min([r for r in res[i + 1:] if r != -1])
        else:
            res[i] = min(res[i], min([r for r in res[i + 1:] if r != -1], default=res[i]))

    idx = 0
    for a in A:
        if res[idx] < a:
            print(res[idx])
            idx += 1
        else:
            print(res[idx])
 
    for i, x in AB:
        i -= 1
        A[i] = x

        if A[i] < res[idx]:
            if A[i] < min(res[i + 1:]):
                res[idx] = A[i]
            else:
                res[idx] = min([r for r in res[i + 1:] if r != -1])
        else:
            res[idx] = A[i]
        print(res[idx])
        idx += 1
    
resolve()