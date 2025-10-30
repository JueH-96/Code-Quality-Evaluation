import sys
import math

def main():
    n, *rest = map(int, sys.stdin.read().split())
    a = rest[:n]
    
    if n == 0:
        print(0)
        return
    
    # Precompute L and R arrays
    L = [a[i] - i for i in range(n)]
    R = [a[i] + i for i in range(n)]
    
    # Build sparse tables for L and R
    def build_sparse_table(arr):
        size = len(arr)
        if size == 0:
            return []
        logn = math.floor(math.log2(size)) + 1
        st = []
        st.append(arr.copy())
        for j in range(1, logn):
            prev = st[j-1]
            current_level = []
            length = 1 << j
            for i in range(size - length + 1):
                current_level.append(min(prev[i], prev[i + (1 << (j-1))]))
            st.append(current_level)
        return st
    
    st_L = build_sparse_table(L)
    st_R = build_sparse_table(R)
    
    # Precompute log_table for range queries
    max_n = n
    log_table = [0] * (max_n + 1)
    for i in range(2, max_n + 1):
        log_table[i] = log_table[i // 2] + 1
    
    def query_min(st, log_table, l, r):
        if l > r:
            return float('inf')
        length = r - l + 1
        k = log_table[length]
        if (1 << k) > length:
            k -= 1
        offset = 1 << k
        return min(st[k][l], st[k][r - offset + 1])
    
    max_k = 0
    for c in range(n):
        possible_k = min(c + 1, n - c)
        low, high = 1, possible_k
        best = 0
        while low <= high:
            mid = (low + high) // 2
            left_steps = mid - 1
            left_start = c - left_steps
            left_end = c
            right_end = c + left_steps
            if left_start < 0 or right_end >= n:
                high = mid - 1
                continue
            # Check left min
            min_l = query_min(st_L, log_table, left_start, left_end)
            if min_l < (mid - c):
                high = mid - 1
                continue
            # Check right min
            min_r = query_min(st_R, log_table, c, right_end)
            if min_r < (mid + c):
                high = mid - 1
            else:
                best = mid
                low = mid + 1
        if best > max_k:
            max_k = best
    print(max_k)

if __name__ == '__main__':
    main()