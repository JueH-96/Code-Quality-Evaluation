from typing import List

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness values in descending order.
        # The idea is that if a child is selected in turn i (starting from 0),
        # its effective happiness is max(happiness - i, 0).
        # To maximize the total sum, we want the largest happiness values to suffer the smallest penalty.
        happiness.sort(reverse=True)
        
        # Calculate the sum over the first k children with appropriate decrement.
        total = 0
        for i in range(k):
            # effective happiness = original - i, at least 0.
            total += max(happiness[i] - i, 0)
        return total

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.maximumHappinessSum([1, 2, 3], 2))  # Output: 4
    # Example 2:
    print(sol.maximumHappinessSum([1, 1, 1, 1], 2))  # Output: 1
    # Example 3:
    print(sol.maximumHappinessSum([2, 3, 4, 5], 1))  # Output: 5