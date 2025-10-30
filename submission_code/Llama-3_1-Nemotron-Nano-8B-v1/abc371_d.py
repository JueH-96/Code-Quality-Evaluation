import sys
import bisect

def main():
    n = int(sys.stdin.readline())
    X = list(map(int, sys.stdin.readline().split()))
    P = list(map(int, sys.stdin.readline().split()))
    
    prefix = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i-1] + P[i-1]
    
    q = int(sys.stdin.readline())
    for _ in range(q):
        L, R = map(int, sys.stdin.readline().split())
        left = bisect.bisect_left(X, L)
        right_pos = bisect.bisect_right(X, R) - 1
        
        if left > right_pos:
            print(0)
        else:
            print(prefix[right_pos + 1] - prefix[left])

if __name__ == "__main__":
    main()