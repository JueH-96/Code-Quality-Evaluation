def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    S = sum(A)
    low = 1
    high = S // K
    best = 0
    
    def compute_max_partitions(arr, x):
        count = 0
        current = 0
        for num in arr:
            current += num
            if current >= x:
                count += 1
                current = 0
        return count
    
    def feasible(x):
        max_linear = compute_max_partitions(A, x)
        if max_linear >= K:
            return True
        
        # Check suffix case
        suffix_sums = [0] * (N + 1)
        for i in range(1, N + 1):
            suffix_sums[i] = suffix_sums[i-1] + A[N - i]
        
        for i in range(1, N + 1):
            if suffix_sums[i] >= x:
                remaining = A[:N - i]
                count_remaining = compute_max_partitions(remaining, x)
                if 1 + count_remaining >= K:
                    return True
        
        # Check prefix case
        prefix_sums = [0] * (N + 1)
        for i in range(1, N + 1):
            prefix_sums[i] = prefix_sums[i-1] + A[i-1]
        
        for i in range(1, N + 1):
            if prefix_sums[i] >= x:
                remaining = A[i:]
                count_remaining = compute_max_partitions(remaining, x)
                if 1 + count_remaining >= K:
                    return True
        
        return False
    
    while low <= high:
        mid = (low + high) // 2
        if feasible(mid):
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    
    # Calculate y
    y = 0
    for i in range(N):
        next_i = (i + 1) % N
        if A[i] + A[next_i] < best:
            y += 1
    
    print(best, y)

if __name__ == '__main__':
    main()