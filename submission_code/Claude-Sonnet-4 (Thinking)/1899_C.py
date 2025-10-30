def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    max_sum = a[0]
    current_sum = a[0]
    
    for i in range(1, n):
        # Check if parity alternates with the previous element
        if a[i-1] % 2 != a[i] % 2:
            # We can extend the current subarray
            current_sum = max(a[i], current_sum + a[i])
        else:
            # We must start a new subarray
            current_sum = a[i]
        
        max_sum = max(max_sum, current_sum)
    
    return max_sum

t = int(input())
for _ in range(t):
    print(solve())