class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in ascending order
        for row in nums:
            row.sort()
        
        # The maximum number of elements across any row
        max_len = max(len(row) for row in nums)
        score = 0
        
        # For each possible removal round (from largest elements to smallest)
        for i in range(max_len):
            removed = []
            for row in nums:
                # If this row still has an element to remove
                if len(row) > i:
                    # The ith largest element is at index -(i+1)
                    removed.append(row[-(i+1)])
            # Add the maximum of the removed elements to the score
            score += max(removed)
        
        return score