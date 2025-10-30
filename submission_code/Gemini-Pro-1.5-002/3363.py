class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        n = len(nums)
        ans = []
        counts = {}
        for i in range(n):
            num = nums[i]
            f = freq[i]
            counts[num] = counts.get(num, 0) + f
            
            if counts[num] == 0:
                del counts[num]
            
            if not counts:
                ans.append(0)
            else:
                max_freq = 0
                for num, count in counts.items():
                    max_freq = max(max_freq, count)
                ans.append(max_freq)
        return ans