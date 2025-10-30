class Solution:
    _min_prime = None

    def minOperations(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        if Solution._min_prime is None:
            max_prime = 10**6 + 1
            min_prime = [0] * max_prime
            min_prime[0], min_prime[1] = 0, 1  # Handle 0 and 1 cases
            
            for i in range(2, max_prime):
                if min_prime[i] == 0:
                    min_prime[i] = i
                    j = i * 2
                    while j < max_prime:
                        if min_prime[j] == 0:
                            min_prime[j] = i
                        j += i
            Solution._min_prime = min_prime
        
        # Generate the options list for each number in nums
        options_list = []
        for num in nums:
            if num == 1:
                options = {1: 0}
            else:
                mp = Solution._min_prime[num]
                if mp == num:
                    # num is a prime
                    options = {num: 0}
                else:
                    reduced = mp
                    options = {num: 0, reduced: 1}
            options_list.append(options)
        
        # Initialize the DP for the first element
        prev_dp = {}
        for key, val in options_list[0].items():
            prev_dp[key] = val
        
        # Process the rest of the elements
        for i in range(1, len(nums)):
            curr_options = options_list[i]
            curr_dp = {}
            
            for prev_val, prev_cost in prev_dp.items():
                for curr_val, curr_cost in curr_options.items():
                    if prev_val <= curr_val:
                        total_cost = prev_cost + curr_cost
                        if curr_val not in curr_dp or total_cost < curr_dp[curr_val]:
                            curr_dp[curr_val] = total_cost
            
            if not curr_dp:
                return -1
            prev_dp = curr_dp
        
        return min(prev_dp.values())