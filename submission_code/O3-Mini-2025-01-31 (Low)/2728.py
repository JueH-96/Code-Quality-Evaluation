class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # For each row, sort the row in ascending order.
        for row in nums:
            row.sort()
        
        # Initialize score.
        score = 0

        # The number of rounds equals the length of the shortest row (or each row, since they are fixed).
        # However, in this problem every row may have different lengths.
        # But note: when rows become empty they should be skipped.
        # A neat trick: we can simulate removal by taking from the end since rows are sorted in ascending order.
        # But since rows can have different lengths, we determine the maximum rounds possible as the maximum 
        # row length. However, in a given round, only rows that still have elements are considered.
        # So we loop until at least one row remains non-empty.
        
        # Alternatively, since removal happens row by row (each row gives exactly one removed element)
        # we can simulate the rounds until all rows are depleted.
        while any(row for row in nums):
            current_round = []
            for row in nums:
                if row:
                    # remove and append the largest element from the row (last element in sorted row)
                    current_round.append(row.pop())
            # add the maximum among these removed elements to the score
            score += max(current_round)
            
        return score