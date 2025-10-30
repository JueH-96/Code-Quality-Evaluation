class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        count = {}
        ans = []
        total_count = 0
        for i in range(len(nums)):
            num = nums[i]
            f = freq[i]
            if num not in count:
                count[num] = 0
            count[num] += f
            total_count += f
            if total_count == 0:
                ans.append(0)
            else:
                max_freq = 0
                for c in count:
                    if count[c] > max_freq:
                        max_freq = count[c]
                ans.append(max_freq)
        return ans