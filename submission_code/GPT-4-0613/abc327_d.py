import sys
from collections import defaultdict

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    d = defaultdict(list)
    for i in range(M):
        if A[i] == B[i]:
            print("No")
            return
        if A[i] > B[i]:
            A[i], B[i] = B[i], A[i]
        d[A[i]].append(B[i])
    for k in sorted(d.keys()):
        d[k].sort()
        if len(d[k]) > 1 and d[k][0] <= k:
            print("No")
            return
        if len(d[k]) > 2 and d[k][1] <= k:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()