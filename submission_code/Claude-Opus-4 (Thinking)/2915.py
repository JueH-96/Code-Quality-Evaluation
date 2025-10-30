class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        prefix_count = {0: 1}  # prefix[0] % modulo = 0 occurs once
        prefix_sum = 0
        result = 0
        
        for num in nums:
            # Update prefix sum
            if num % modulo == k:
                prefix_sum += 1
            
            # We want (prefix_sum - prefix[l]) % modulo == k
            # So prefix[l] % modulo == (prefix_sum - k) % modulo
            target = (prefix_sum - k) % modulo
            if target in prefix_count:
                result += prefix_count[target]
            
            # Add current prefix_sum % modulo to the count
            prefix_sum_mod = prefix_sum % modulo
            prefix_count[prefix_sum_mod] = prefix_count.get(prefix_sum_mod, 0) + 1
        
        return result