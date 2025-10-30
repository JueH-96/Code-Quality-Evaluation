def minimumValueSum(nums, andValues):
    n = len(nums)
    m = len(andValues)
    
    if m > n:
        return -1
    
    # Precompute all possible ANDs for subarrays ending at i
    # We'll use a list of dictionaries where for each i, we track possible AND values and their minimal sums
    # For each k, we'll process the array to track the minimal sum for exactly k subarrays ending at each i
    dp = [{} for _ in range(m + 1)]
    
    for k in range(1, m + 1):
        A = andValues[k - 1]
        current_dp = {}
        for i in range(n):
            # Initialize with no subarrays ending before i
            if k == 0:
                current_dp[A] = 0
                continue
            # Reset current_dp for this i
            current_dp = {}
            # Consider all possible j < i
            for j in range(i):
                # Compute the AND from j+1 to i
                current_and = nums[i]
                for t in range(j, i):
                    current_and &= nums[t]
                if current_and == A:
                    # If we can split into k-1 subarrays up to j, then adding this subarray gives k subarrays
                    if k - 1 <= m:
                        if j == 0:
                            prev_sum = 0 if k == 1 else float('inf')
                        else:
                            prev_sum = dp[k-1].get(j, float('inf'))
                        if prev_sum != float('inf'):
                            total = prev_sum + nums[i]
                            if A in current_dp:
                                current_dp[A] = min(current_dp[A], total)
                            else:
                                current_dp[A] = total
            # Also, check if the subarray is from i itself (only if k=1)
            if k == 1 and nums[i] == A:
                if A in current_dp:
                    current_dp[A] = min(current_dp[A], nums[i])
                else:
                    current_dp[A] = nums[i]
            # Update dp for k
            if current_dp:
                dp[k] = current_dp.copy()
            else:
                # No way to split into k subarrays ending at i
                dp[k] = {}
    
    # After processing all k, check if the last element can be split into m subarrays
    if m <= len(dp) and dp[m]:
        # The minimal sum is the minimal value in dp[m] for the last element
        # But wait, dp[m] is a dictionary of possible AND values and their sums
        # Since the last subarray must end at n-1, we need to look for the entry where the AND is A and the position is n-1
        # But in our setup, dp[k][i] is the minimal sum for k subarrays ending at i
        # So, after processing all i, we need to look for the minimal sum in dp[m] where the key is the AND value (which is A)
        # Because the last subarray must have an AND of A
        if A in dp[m]:
            return dp[m][A]
        else:
            return -1
    else:
        return -1

# Example usage
# nums = [1,4,3,3,2]
# andValues = [0,3,3,2]
# print(minimumValueSum(nums, andValues))