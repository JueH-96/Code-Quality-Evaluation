import sys

def solve():
    N = int(sys.stdin.readline().strip())
    H = list(map(int, sys.stdin.readline().strip().split()))

    T = 0
    while any(h > 0 for h in H):
        T += 1
        for i in range(N):
            if H[i] > 0:
                H[i] -= 1
            else:
                H[i] -= 1
            if T % 3 == 0 and H[i] > 0:
                H[i] -= 3
        if T % 3 == 0:
            H = [h-3 if h-3 >= 0 else 0 for h in H]
        else:
            H = [h-1 if h-1 >= 0 else 0 for h in H]

    print(T)

solve()