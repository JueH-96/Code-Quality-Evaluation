import bisect

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        pals = []
        for d in range(1, 10):
            start = 10**((d-1)//2)
            end = 10**((d+1)//2)
            for half in range(start, end):
                s = str(half)
                if d % 2 == 1:
                    palindrome_str = s + s[:-1][::-1]
                else:
                    palindrome_str = s + s[::-1]
                pal_num = int(palindrome_str)
                pals.append(pal_num)
        
        n = len(nums)
        sorted_nums = sorted(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + sorted_nums[i]
        
        min_cost = float('inf')
        total_sum = prefix[n]
        
        for pal in pals:
            idx = bisect.bisect_left(sorted_nums, pal)
            left_sum = pal * idx - prefix[idx]
            right_sum = (total_sum - prefix[idx]) - pal * (n - idx)
            cost = left_sum + right_sum
            if cost < min_cost:
                min_cost = cost
        
        return min_cost