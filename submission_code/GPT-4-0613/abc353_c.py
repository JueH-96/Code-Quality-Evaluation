import sys
from itertools import combinations

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    mod = 10**8
    ans = 0
    for i in range(N):
        for j in range(i+1, N):
            ans += (A[i] + A[j]) % mod
    print(ans)

if __name__ == "__main__":
    main()