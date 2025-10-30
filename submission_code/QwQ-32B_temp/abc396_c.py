import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    B = list(map(int, sys.stdin.readline().split()))
    W = list(map(int, sys.stdin.readline().split()))
    
    # Sort the balls in descending order
    B.sort(reverse=True)
    W.sort(reverse=True)
    
    # Compute prefix sums for black balls
    prefix_b = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_b[i] = prefix_b[i - 1] + B[i - 1]
    
    # Compute prefix sums for white balls
    prefix_w = [0] * (m + 1)
    for i in range(1, m + 1):
        prefix_w[i] = prefix_w[i - 1] + W[i - 1]
    
    # Compute max_B array
    max_b = [0] * (n + 1)
    max_b[n] = prefix_b[n]
    for k in range(n - 1, -1, -1):
        max_b[k] = max(prefix_b[k], max_b[k + 1])
    
    max_total = -float('inf')
    min_val = min(n, m)
    for w in range(0, min_val + 1):
        current = prefix_w[w] + max_b[w]
        if current > max_total:
            max_total = current
    
    print(max_total)

if __name__ == "__main__":
    main()