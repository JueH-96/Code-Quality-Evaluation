class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness values in descending order
        happiness.sort(reverse=True)
        
        # We will pick the child with the largest remaining happiness first,
        # then the second largest, and so on.
        # If we pick a child at turn i (0-indexed), that child's happiness is reduced by i
        # (but not below zero). Summing up max(h - i, 0) for the top k children
        # in the sorted list gives the maximum total happiness.
        
        total = 0
        for i in range(k):
            # Calculate the effective happiness for pick number i
            val = happiness[i] - i
            if val > 0:
                total += val
        
        return total