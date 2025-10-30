class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        pre = 0
        min_pre = {}
        max_sum = -10**18
        
        for num in nums:
            temp = pre
            pre += num
            targets = [num + k, num - k]
            for target in targets:
                if target in min_pre:
                    current_sum = pre - min_pre[target]
                    if current_sum > max_sum:
                        max_sum = current_sum
            if num not in min_pre or temp < min_pre[num]:
                min_pre[num] = temp
        
        return max_sum if max_sum != -10**18 else 0