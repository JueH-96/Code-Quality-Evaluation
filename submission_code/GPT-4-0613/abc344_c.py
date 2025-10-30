import bisect

N = int(input())
A = list(map(int, input().split()))
M = int(input())
B = list(map(int, input().split()))
L = int(input())
C = list(map(int, input().split()))
Q = int(input())
X = list(map(int, input().split()))

A.sort()
B.sort()
C.sort()

for x in X:
    for a in A:
        if a > x:
            break
        for b in B:
            if a + b > x:
                break
            if bisect.bisect_left(C, x - a - b) != L and C[bisect.bisect_left(C, x - a - b)] == x - a - b:
                print("Yes")
                break
        else:
            continue
        break
    else:
        print("No")