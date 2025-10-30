class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        for row in nums:
            row.sort(reverse=True)
        total = 0
        max_len = max(len(row) for row in nums) if nums else 0
        for i in range(max_len):
            current_max = 0
            for row in nums:
                if i < len(row):
                    if row[i] > current_max:
                        current_max = row[i]
            total += current_max
        return total