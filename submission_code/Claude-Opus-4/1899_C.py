def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    max_sum = float('-inf')
    
    # For each starting position
    for i in range(n):
        current_sum = a[i]
        max_sum = max(max_sum, current_sum)
        
        # Try to extend the subarray
        for j in range(i + 1, n):
            # Check if parities alternate
            if (a[j-1] % 2) != (a[j] % 2):
                current_sum += a[j]
                max_sum = max(max_sum, current_sum)
            else:
                # Can't extend further with alternating parities
                break
    
    return max_sum

t = int(input())
for _ in range(t):
    print(solve())