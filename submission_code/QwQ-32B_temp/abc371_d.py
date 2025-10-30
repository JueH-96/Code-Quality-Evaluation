import bisect
import sys

def main():
    n = int(sys.stdin.readline())
    X = list(map(int, sys.stdin.readline().split()))
    P = list(map(int, sys.stdin.readline().split()))
    
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + P[i]
    
    q = int(sys.stdin.readline())
    for _ in range(q):
        L, R = map(int, sys.stdin.readline().split())
        left = bisect.bisect_left(X, L)
        right = bisect.bisect_right(X, R) - 1
        if left > right:
            print(0)
        else:
            print(prefix[right + 1] - prefix[left])
            
if __name__ == "__main__":
    main()