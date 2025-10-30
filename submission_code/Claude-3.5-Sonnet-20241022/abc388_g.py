def max_kagamimochi(arr, l, r):
    # Get the subarray for current query
    sub = arr[l-1:r]
    n = len(sub)
    
    # Try each possible number of pairs
    left = 0
    right = n // 2
    
    # Binary search on the number of pairs
    result = 0
    while left <= right:
        mid = (left + right) // 2
        
        # Check if we can make 'mid' number of pairs
        # Using two pointers approach
        possible = False
        used = [False] * n
        count = 0
        
        # Try to find pairs starting from smallest numbers
        i = 0  # smaller number
        j = 0  # larger number
        
        while i < n and count < mid:
            # Find next unused smaller number
            while i < n and used[i]:
                i += 1
            if i >= n:
                break
                
            # Find next unused larger number that satisfies the condition
            j = i + 1
            found = False
            while j < n and not found:
                if not used[j] and sub[i] * 2 <= sub[j]:
                    used[i] = True
                    used[j] = True
                    count += 1
                    found = True
                j += 1
                
            if not found:
                break
            i += 1
            
        if count == mid:
            possible = True
            
        if possible:
            result = mid
            left = mid + 1
        else:
            right = mid - 1
            
    return result

# Read input
N = int(input())
A = list(map(int, input().split()))
Q = int(input())

# Process queries
for _ in range(Q):
    L, R = map(int, input().split())
    result = max_kagamimochi(A, L, R)
    print(result)