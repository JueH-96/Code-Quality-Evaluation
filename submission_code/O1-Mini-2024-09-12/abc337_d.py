# YOUR CODE HERE
import sys

def main():
    import sys
    import math
    H, W, K = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]
    
    INF = math.inf
    min_ops = INF
    
    # Process rows
    for row in S:
        segments = row.split('x')
        for seg in segments:
            L = len(seg)
            if L < K:
                continue
            # Convert to list of 1 if '.', 0 if 'o'
            dots = [1 if c == '.' else 0 for c in seg]
            # Compute prefix sums
            prefix = [0] * (L + 1)
            for i in range(L):
                prefix[i+1] = prefix[i] + dots[i]
            # Slide window
            for i in range(L - K + 1):
                num_dots = prefix[i+K] - prefix[i]
                min_ops = min(min_ops, num_dots)
    
    # Process columns
    for col in range(W):
        column = ''.join(S[row][col] for row in range(H))
        segments = column.split('x')
        for seg in segments:
            L = len(seg)
            if L < K:
                continue
            # Convert to list of 1 if '.', 0 if 'o'
            dots = [1 if c == '.' else 0 for c in seg]
            # Compute prefix sums
            prefix = [0] * (L + 1)
            for i in range(L):
                prefix[i+1] = prefix[i] + dots[i]
            # Slide window
            for i in range(L - K + 1):
                num_dots = prefix[i+K] - prefix[i]
                min_ops = min(min_ops, num_dots)
    
    print(-1 if min_ops == INF else min_ops)

if __name__ == "__main__":
    main()