class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        count = 0
        prefix_count = {0: 1}  # To handle the case where cnt % modulo == k directly
        cnt = 0
        
        for num in nums:
            if num % modulo == k:
                cnt += 1
            
            # We need to check if (cnt - k) % modulo is in prefix_count
            target = (cnt - k) % modulo
            
            if target in prefix_count:
                count += prefix_count[target]
            
            # Update the prefix_count for the current cnt % modulo
            current_mod = cnt % modulo
            if current_mod in prefix_count:
                prefix_count[current_mod] += 1
            else:
                prefix_count[current_mod] = 1
        
        return count