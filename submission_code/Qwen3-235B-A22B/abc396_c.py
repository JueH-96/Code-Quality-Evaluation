import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    B = list(map(int, data[idx:idx+N]))
    idx += N
    W = list(map(int, data[idx:idx+M]))
    idx += M
    
    B.sort(reverse=True)
    W.sort(reverse=True)
    
    # Compute prefix sums for black balls
    prefix_b = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_b[i] = prefix_b[i - 1] + B[i - 1]
    
    # Compute prefix sums for white balls
    prefix_w = [0] * (M + 1)
    for i in range(1, M + 1):
        prefix_w[i] = prefix_w[i - 1] + W[i - 1]
    
    # Compute max_white array
    max_white = [0] * (M + 1)
    current_max = 0
    for i in range(1, M + 1):
        current_max = max(current_max, prefix_w[i])
        max_white[i] = current_max
    
    # Find the maximum sum
    max_total = 0
    for b in range(0, N + 1):
        allowed_white = min(b, M)
        current_sum = prefix_b[b] + max_white[allowed_white]
        if current_sum > max_total:
            max_total = current_sum
    
    print(max_total)

if __name__ == "__main__":
    main()