from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        groups = defaultdict(list)
        for num in nums:
            s = str(num)
            max_digit = max(s)
            groups[max_digit].append(num)
        
        ans = -1
        for key in groups:
            arr = groups[key]
            if len(arr) >= 2:
                arr.sort()
                total = arr[-1] + arr[-2]
                if total > ans:
                    ans = total
        return ans