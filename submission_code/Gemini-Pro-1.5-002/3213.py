class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                max_val = 0
                for num in sub_array:
                    max_val = max(max_val, num)
                
                count = 0
                for num in sub_array:
                    if num == max_val:
                        count += 1
                
                if count >= k:
                    ans += 1
        return ans