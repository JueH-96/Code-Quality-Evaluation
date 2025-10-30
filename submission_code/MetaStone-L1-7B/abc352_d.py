import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    
    pos = [0] * (N + 2)  # 1-based indexing
    for i in range(N):
        pos[P[i]] = i + 1  # positions are 1-based
    
    # Compute log_table
    log_table = [0] * (N + 2)
    for i in range(2, N + 1):
        log_table[i] = log_table[i // 2] + 1
    max_level = log_table[N] + 1 if N > 0 else 0
    
    # Build max and min tables
    max_table = [[0] * (max_level + 1) for _ in range(N + 2)]
    min_table = [[0] * (max_level + 1) for _ in range(N + 2)]
    
    for i in range(1, N + 1):
        max_table[i][0] = pos[i]
        min_table[i][0] = pos[i]
    
    for k in range(1, max_level + 1):
        for i in range(1, N + 1):
            j = i + (1 << (k - 1))
            if j > N:
                continue
            max_table[i][k] = max(max_table[i][k-1], max_table[j][k-1])
            min_table[i][k] = min(min_table[i][k-1], min_table[j][k-1])
    
    # Functions to get min and max in [l, r]
    def get_min(l, r):
        if l > r:
            return 0
        length = r - l + 1
        k = log_table[length]
        a = min_table[l][k]
        b = min_table[r - (1 << k) + 1][k]
        return min(a, b)
    
    def get_max(l, r):
        if l > r:
            return 0
        length = r - l + 1
        k = log_table[length]
        a = max_table[l][k]
        b = max_table[r - (1 << k) + 1][k]
        return max(a, b)
    
    min_span = float('inf')
    for a in range(1, N - K + 2):
        l = a
        r = a + K - 1
        current_min = get_min(l, r)
        current_max = get_max(l, r)
        current_span = current_max - current_min
        if current_span < min_span:
            min_span = current_span
    
    print(min_span)

if __name__ == '__main__':
    main()