from bisect import bisect_left, bisect_right

N = int(input())
A = sorted([int(x) for x in input().split()])
M = int(input())
B = [int(x) for x in input().split()]
L = int(input())
C = sorted([int(x) for x in input().split()])
Q = int(input())
X = [int(x) for x in input().split()]

for q in range(Q):
    x = X[q]
    found = False
    for a in A:
        if a > x: break
        for b in B:
            remaining = x - a - b
            left = bisect_left(C, remaining)
            right = bisect_right(C, remaining)
            if left < right:
                found = True
                break
        if found: break
    if found:
        print("Yes")
    else:
        print("No")