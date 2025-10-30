class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        count = 0
        prefix_sum = 0
        prefix_count = {0: 1}
        
        for num in nums:
            if num % modulo == k:
                prefix_sum += 1
            target = (prefix_sum - k) % modulo
            if target in prefix_count:
                count += prefix_count[target]
            prefix_count[prefix_sum % modulo] = prefix_count.get(prefix_sum % modulo, 0) + 1
        
        return count