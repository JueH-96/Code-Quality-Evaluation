from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        prefix_count = 0
        count = defaultdict(int)
        count[0] = 1
        result = 0
        
        for num in nums:
            prefix_count += num % modulo == k
            result += count[(prefix_count - k) % modulo]
            count[prefix_count % modulo] += 1
        
        return result