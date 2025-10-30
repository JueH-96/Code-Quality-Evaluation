def count_arithmetic_progressions(arr, n):
    count = 0
    
    for l in range(n):
        # Every sequence of length 1 is an arithmetic progression
        count += 1
        
        if l + 1 < n:
            diff = arr[l+1] - arr[l]
            
            # Sequences of length 2 are always arithmetic progressions
            count += 1
            
            # Try to extend the sequence
            for r in range(l+2, n):
                if arr[r] - arr[r-1] == diff:
                    count += 1
                else:
                    break
    
    return count

# Read input
N = int(input())
A = list(map(int, input().split()))

# Solve and print result
print(count_arithmetic_progressions(A, N))