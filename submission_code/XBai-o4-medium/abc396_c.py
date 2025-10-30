import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    B = list(map(int, sys.stdin.readline().split()))
    W = list(map(int, sys.stdin.readline().split()))
    
    B.sort(reverse=True)
    W.sort(reverse=True)
    
    # Compute prefix sums for black balls
    prefix_black = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_black[i] = prefix_black[i-1] + B[i-1]
    
    # Compute prefix sums for white balls
    prefix_white = [0] * (m + 1)
    for j in range(1, m + 1):
        prefix_white[j] = prefix_white[j-1] + W[j-1]
    
    # Compute max_suffix for black prefix sums
    max_suffix = [0] * (n + 2)  # Extra space to avoid index issues
    max_suffix[n] = prefix_black[n]
    for i in range(n-1, -1, -1):
        max_suffix[i] = max(prefix_black[i], max_suffix[i+1])
    
    possible_w_max = min(n, m)
    max_total = -float('inf')
    
    for w in range(0, possible_w_max + 1):
        current = prefix_white[w] + max_suffix[w]
        if current > max_total:
            max_total = current
    
    print(max_total)

if __name__ == "__main__":
    main()