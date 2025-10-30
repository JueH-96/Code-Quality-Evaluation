class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        prefix_ok = [True] * n
        for i in range(1, n):
            prefix_ok[i] = prefix_ok[i-1] and (nums[i] > nums[i-1])
        
        suffix_ok = [True] * n
        for i in range(n-2, -1, -1):
            suffix_ok[i] = (nums[i] < nums[i+1]) and suffix_ok[i+1]
        
        count = 0
        for l in range(n):
            for r in range(l, n):
                if (l == 0 or prefix_ok[l-1]) and (r == n-1 or suffix_ok[r+1]) and (l == 0 or r == n-1 or nums[l-1] < nums[r+1]):
                    count += 1
        return count