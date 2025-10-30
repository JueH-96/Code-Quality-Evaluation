def solve():
    B = int(input())
    
    # For small values of A, check if A^A equals B
    for A in range(1, 100):
        if A**A == B:
            return A
            
    # For larger values, use binary search
    # Since A^A grows very fast, we can limit our search
    left = 1
    right = min(10**18, B)
    
    while left <= right:
        mid = (left + right) // 2
        val = mid**mid
        
        if val == B:
            return mid
        elif val < B:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1

print(solve())