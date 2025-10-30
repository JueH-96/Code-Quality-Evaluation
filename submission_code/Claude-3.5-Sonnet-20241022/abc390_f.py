def min_operations(arr, L, R):
    # Get the subarray from L to R (0-based indexing)
    sub = arr[L:R+1]
    
    # Count frequency of each number
    freq = {}
    for x in sub:
        freq[x] = freq.get(x, 0) + 1
    
    # Get unique numbers in sorted order
    nums = sorted(set(sub))
    n = len(nums)
    
    # dp[mask] = minimum operations needed for remaining numbers represented by mask
    dp = {}
    
    def solve(mask):
        if mask == 0:
            return 0
        if mask in dp:
            return dp[mask]
        
        # Try all possible ranges [l,r] to erase
        result = float('inf')
        for l in range(n):
            if not (mask & (1 << l)):  # if lth number is already erased
                continue
            # Try all possible r >= l
            curr_mask = 0
            for r in range(l, n):
                if not (mask & (1 << r)):  # if rth number is already erased
                    continue
                # Add this number to current range
                curr_mask |= (1 << r)
                # Check if we can erase this range
                valid = True
                for i in range(nums[l], nums[r] + 1):
                    if i not in freq:  # number not in original array
                        valid = False
                        break
                if valid:
                    # Calculate remaining mask after erasing current range
                    next_mask = mask & ~curr_mask
                    result = min(result, 1 + solve(next_mask))
            
        dp[mask] = result
        return result
    
    return solve((1 << n) - 1)

def solve():
    # Read input
    N = int(input())
    A = list(map(int, input().split()))
    
    # Calculate sum of f(L,R) for all L,R
    total = 0
    for L in range(N):
        for R in range(L, N):
            total += min_operations(A, L, R)
    
    print(total)

# Run the solution
solve()