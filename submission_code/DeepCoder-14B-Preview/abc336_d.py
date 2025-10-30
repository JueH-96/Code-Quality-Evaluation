def build_sparse_table(arr):
    n = len(arr)
    if n == 0:
        return None, None
    log_table = [0] * (n + 1)
    for i in range(2, n + 1):
        log_table[i] = log_table[i // 2] + 1
    k = log_table[n] if n > 0 else 0
    st = [ [0] * n for _ in range(k + 1) ]
    st[0] = arr[:]
    for j in range(1, k + 1):
        for i in range(n - (1 << j) + 1):
            st[j][i] = min(st[j-1][i], st[j-1][i + (1 << (j-1))])
    return st, log_table

def range_min(st, log_table, L, R):
    if L > R:
        return float('inf')
    length = R - L + 1
    j = log_table[length]
    return min(st[j][L], st[j][ R - (1 << j) + 1 ])

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    a = list(map(int, data[1:n+1]))
    
    if n == 0:
        print(0)
        return
    
    left = [a[i] - i for i in range(n)]
    right = [a[i] + i for i in range(n)]
    
    left_st, left_log = build_sparse_table(left)
    right_st, right_log = build_sparse_table(right)
    
    def is_possible(k):
        window_size = 2 * k - 1
        if window_size > n:
            return False
        max_s = n - window_size
        for s in range(max_s + 1):
            end_first = s + k - 1
            if end_first >= n:
                continue
            min_left = range_min(left_st, left_log, s, end_first)
            if min_left < (1 - s):
                continue
            start_second = s + k
            end_second = s + 2 * k - 2
            if start_second > end_second:
                return True
            if end_second >= n:
                continue
            min_right = range_min(right_st, right_log, start_second, end_second)
            if min_right < (2 * k + s - 1):
                continue
            return True
        return False
    
    low = 1
    high = (n + 1) // 2
    best = 0
    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            continue
        if is_possible(mid):
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    print(best)

if __name__ == "__main__":
    main()