class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        starts = [i for i, num in enumerate(nums) if num == 0]
        count = 0
        n = len(nums)
        for start in starts:
            for direction in (-1, 1):
                arr = list(nums)  # Make a copy of the original array
                curr = start
                d = direction
                while True:
                    if curr < 0 or curr >= n:
                        break
                    if arr[curr] == 0:
                        curr += d
                    else:
                        arr[curr] -= 1
                        d *= -1  # Reverse direction
                        curr += d
                # Check if all elements are zero
                if all(x == 0 for x in arr):
                    count += 1
        return count