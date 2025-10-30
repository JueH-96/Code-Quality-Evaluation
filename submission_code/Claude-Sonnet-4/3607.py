class Solution:
    def minOperations(self, nums: List[int]) -> int:
        def get_smallest_prime_factor(n):
            if n <= 1:
                return n
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return i
            return n
        
        def get_all_possible_values(n):
            # Get all possible values we can reduce n to
            values = []
            current = n
            while True:
                values.append(current)
                if current == 1:
                    break
                spf = get_smallest_prime_factor(current)
                if spf == current:  # current is prime
                    values.append(1)
                    break
                current = spf
            return values
        
        n = len(nums)
        if n == 1:
            return 0
            
        # For each position, get all possible values and their operation counts
        possible_values = []
        for num in nums:
            values = get_all_possible_values(num)
            possible_values.append([(val, i) for i, val in enumerate(values)])
        
        # DP: dp[i] = minimum operations to make first i+1 elements non-decreasing
        # We need to track the last value used
        from collections import defaultdict
        
        # dp[value] = minimum operations to reach this state
        dp = defaultdict(lambda: float('inf'))
        
        # Initialize with first element
        for val, ops in possible_values[0]:
            dp[val] = ops
        
        # Process remaining elements
        for i in range(1, n):
            new_dp = defaultdict(lambda: float('inf'))
            
            for curr_val, curr_ops in possible_values[i]:
                # Try to use curr_val at position i
                for prev_val, prev_total_ops in dp.items():
                    if prev_val <= curr_val:  # Non-decreasing condition
                        new_dp[curr_val] = min(new_dp[curr_val], prev_total_ops + curr_ops)
            
            if not new_dp:  # No valid transitions possible
                return -1
                
            dp = new_dp
        
        return min(dp.values()) if dp else -1