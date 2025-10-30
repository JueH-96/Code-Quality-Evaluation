class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        b = [0] * n  # b[i] stores the number of operations starting at index i

        for i in range(n):
            decrease = 0
            for j in range(max(0, i - k + 1), i):
                decrease += b[j]

            if nums[i] - decrease < 0:
                return False  # Cannot make the element 0

            if nums[i] - decrease > 0:
                operations = nums[i] - decrease
                if i + k > n:
                    return False
                b[i] = operations

        # Verify if all elements can be made 0
        for i in range(n):
            total_decrease = 0
            for j in range(max(0, i - k + 1), min(n, i + 1)):
                total_decrease += b[j]
            if nums[i] != total_decrease:
                return False

        return True