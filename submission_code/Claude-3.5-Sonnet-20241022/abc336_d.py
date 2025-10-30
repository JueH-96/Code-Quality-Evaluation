N = int(input())
A = list(map(int, input().split()))

def is_pyramid(arr, k):
    if len(arr) != 2*k-1:
        return False
    for i in range(k-1):
        if arr[i] != i+1 or arr[-i-1] != i+1:
            return False
    if arr[k-1] != k:
        return False
    return True

def can_make_pyramid(arr, k, ops):
    if k == 1:
        return True
        
    target = [i+1 for i in range(k)] + [i for i in range(k-1,0,-1)]
    n = len(arr)
    
    def check(left, right, decrements):
        if len(decrements) != n:
            return False
            
        # Check if we have enough operations
        if sum(decrements) + left + right > ops:
            return False
            
        # Check if decrements are valid
        for i in range(n):
            if decrements[i] > A[i]-1:
                return False
                
        # Get resulting array after decrements
        curr = [A[i] - decrements[i] for i in range(n)]
        
        # Remove left elements
        curr = curr[left:]
        
        # Remove right elements
        curr = curr[:len(curr)-right]
        
        # Check if matches target
        if len(curr) != len(target):
            return False
            
        for i in range(len(curr)):
            if curr[i] != target[i]:
                return False
                
        return True

    # Try all possible combinations of left/right removals
    for left in range(n+1):
        for right in range(n+1):
            if left + right > n:
                continue
            if n - (left + right) != 2*k-1:
                continue
                
            # Try to match remaining elements with pyramid
            decrements = []
            valid = True
            curr_pos = 0
            
            for i in range(left, n-right):
                val = A[i]
                target_val = target[curr_pos]
                if val < target_val:
                    valid = False
                    break
                decrements.append(val - target_val)
                curr_pos += 1
                
            if valid:
                # Fill in decrements for removed elements
                decrements = [0]*left + decrements + [0]*right
                if check(left, right, decrements):
                    return True
                    
    return False

# Binary search on k
left = 1
right = (N+1)//2 + 1

while right - left > 1:
    mid = (left + right) // 2
    ops_needed = 0
    
    # Calculate minimum operations needed
    target = [i+1 for i in range(mid)] + [i for i in range(mid-1,0,-1)]
    
    if can_make_pyramid(A, mid, N*10**9):
        left = mid
    else:
        right = mid
        
print(left)