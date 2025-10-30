import sys
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    A.sort()
    ans = 0
    cnt = defaultdict(int)
    for i in range(N):
        ans += i - cnt[A[i]]
        cnt[A[i]] += 1
    print(ans)

if __name__ == "__main__":
    main()