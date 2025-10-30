class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        total_initial = sum(nums)
        n_val = len(nums)
        zeros = [i for i in range(n_val) if nums[i] == 0]
        count = 0
        for start in zeros:
            for direction in (-1, 1):
                arr = nums.copy()
                total_remaining = total_initial
                curr = start + direction
                d = direction
                n = n_val
                while 0 <= curr < n and total_remaining > 0:
                    if arr[curr] == 0:
                        curr += d
                    else:
                        arr[curr] -= 1
                        total_remaining -= 1
                        d = -d
                        curr += d
                if total_remaining == 0:
                    count += 1
        return count