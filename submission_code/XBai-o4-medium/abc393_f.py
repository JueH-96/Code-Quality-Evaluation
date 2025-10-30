import bisect
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    queries = [[] for _ in range(N+1)]  # queries[R] has list of (X, idx)
    for idx in range(Q):
        R = int(input[ptr])
        ptr += 1
        X = int(input[ptr])
        ptr += 1
        queries[R].append((X, idx))
    
    ans = [0] * Q
    dp = []
    
    for R in range(1, N+1):
        num = A[R-1]
        # Update the dp array for the current number
        i = bisect.bisect_left(dp, num)
        if i < len(dp):
            dp[i] = num
        else:
            dp.append(num)
        # Process all queries for this R
        for (X, idx) in queries[R]:
            pos = bisect.bisect_right(dp, X)
            ans[idx] = pos
    
    for a in ans:
        print(a)

if __name__ == '__main__':
    main()