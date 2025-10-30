import sys
from bisect import bisect_right

def main():
    n, k = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    a.sort()
    l = [0]*(n+1)
    for i in range(n):
        l[i+1] = l[i] + a[i]
    ans = float('inf')
    for i in range(n-k+1):
        ans = min(ans, a[i+k-1]-a[i])
    print(ans)

if __name__ == "__main__":
    main()