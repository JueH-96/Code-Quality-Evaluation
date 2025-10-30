class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        while True:
            removed_numbers = []
            all_rows_empty = True
            for row in nums:
                if row:
                    all_rows_empty = False
                    max_val = -1
                    max_index = -1
                    for i in range(len(row)):
                        if row[i] > max_val:
                            max_val = row[i]
                            max_index = i
                    removed_numbers.append(row.pop(max_index))
            if all_rows_empty:
                break
            if removed_numbers:
                score += max(removed_numbers)
        return score