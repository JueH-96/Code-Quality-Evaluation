class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = nums.length
        count = 0
        
        for i in range(n):
            max_elem = nums[i]
            freq = {}
            for j in range(i, n):
                if nums[j] in freq:
                    freq[nums[j]] += 1
                else:
                    freq[nums[j]] = 1
                
                max_elem = max(max_elem, nums[j])
                
                if freq[max_elem] >= k:
                    count += 1
        
        return count