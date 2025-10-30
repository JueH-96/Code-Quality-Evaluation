from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        def count_at_most(k: int) -> int:
            count = 0
            left = 0
            freq = {}
            distinct = 0
            for right in range(len(nums)):
                if nums[right] not in freq or freq[nums[right]] == 0:
                    distinct +=1
                freq[nums[right]] = freq.get(nums[right], 0) +1
                while distinct >k:
                    freq[nums[left]] -=1
                    if freq[nums[left]] ==0:
                        distinct -=1
                    left +=1
                count += right - left +1
            return count
        
        n = len(nums)
        total = n*(n+1)//2
        median_idx = (total +1)//2
        unique_elements = len(set(nums))
        low, high =1, unique_elements
        while low < high:
            mid = (low + high)//2
            cnt = count_at_most(mid)
            if cnt >= median_idx:
                high = mid
            else:
                low = mid +1
        return low