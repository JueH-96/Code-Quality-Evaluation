def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    
    A = list(map(int, data[ptr:ptr + N]))
    ptr += N
    
    max_val = N + 1
    freq = [0] * (max_val + 1)
    
    for a in A:
        if a <= max_val:
            freq[a] += 1
    
    for _ in range(Q):
        i = int(data[ptr]) - 1  # 0-based index
        ptr += 1
        x = int(data[ptr])
        ptr += 1
        
        old_val = A[i]
        if old_val == x:
            continue
        
        if old_val <= max_val:
            freq[old_val] -= 1
        if x <= max_val:
            freq[x] += 1
        
        A[i] = x
        
        low = 0
        high = max_val
        while low < high:
            mid = (low + high) // 2
            if freq[mid] == 0:
                high = mid
            else:
                low = mid + 1
        
        print(low)

if __name__ == '__main__':
    main()