def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    P = int(data[2])
    
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+N+M]))
    
    # Sort B for binary search
    B_sorted = sorted(B)
    
    total = 0
    
    for a in A:
        # Calculate the maximum b such that a + b < P
        # So b < P - a
        # We need to find the number of b's less than P - a
        # and the sum of those b's
        # Also, for b >= P - a, the sum is P
        
        # Find the index where b >= P - a
        low = 0
        high = M - 1
        idx = M  # default to all b's are less than P - a
        
        while low <= high:
            mid = (low + high) // 2
            if B_sorted[mid] >= P - a:
                idx = mid
                high = mid - 1
            else:
                low = mid + 1
        
        # Sum of b's less than P - a
        sum_less = 0
        if idx > 0:
            sum_less = sum(B_sorted[:idx])
        
        # Number of b's less than P - a
        count_less = idx
        
        # Sum of b's >= P - a
        sum_greater = P * (M - count_less)
        
        # Total for this a
        total += a * count_less + sum_less + sum_greater
    
    print(total)

if __name__ == "__main__":
    main()