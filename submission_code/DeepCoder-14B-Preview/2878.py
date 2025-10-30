class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        if k == 0:
            return all(x == 0 for x in nums)
        max_start = n - k
        add = [0] * (max_start + 1)  # Starting indices are 0 to max_start
        current = 0
        for i in range(n):
            if i >= k:
                current -= add[i - k]
            if i <= max_start:
                required = nums[i] - current
                if required < 0:
                    return False
                add[i] = required
                current += add[i]
            else:
                if current != nums[i]:
                    return False
        return True