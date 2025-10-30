from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the maximum heights in ascending order
        A = sorted(maximumHeight)
        n = len(A)
        # Initialize the next allowable height (one above the last assigned)
        # Start with something larger than any possible A[i]
        h_next = A[-1] + 1
        total = 0
        
        # Assign heights from largest maximum down to smallest
        for i in range(n - 1, -1, -1):
            # We can assign at most A[i], but also must be strictly less than h_next
            h_i = min(A[i], h_next - 1)
            # If we drop below 1, it's impossible to assign distinct positive heights
            if h_i < 1:
                return -1
            total += h_i
            h_next = h_i
        
        return total