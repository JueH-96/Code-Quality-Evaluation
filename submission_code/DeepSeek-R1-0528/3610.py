from typing import List

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        freq = {}
        for i in range(k):
            freq[nums[i]] = freq.get(nums[i], 0) + 1
        
        res = []
        for i in range(n - k + 1):
            arr = [(cnt, num) for num, cnt in freq.items()]
            arr.sort(key=lambda x: (x[0], x[1]), reverse=True)
            s = 0
            for cnt, num in arr[:x]:
                s += cnt * num
            res.append(s)
            
            if i < n - k:
                freq[nums[i]] -= 1
                if freq[nums[i]] == 0:
                    del freq[nums[i]]
                freq[nums[i + k]] = freq.get(nums[i + k], 0) + 1
        
        return res