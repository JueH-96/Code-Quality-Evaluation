class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        max_k = (n + 2) // 3  # Calculate the maximum possible k
        for k in range(0, max_k + 1):
            sub = nums[3 * k:]
            if len(sub) == 0:
                return k
            if len(set(sub)) == len(sub):
                return k
        return max_k  # This line is theoretically unreachable