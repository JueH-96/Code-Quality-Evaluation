class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness values in descending order
        happiness.sort(reverse=True)
        
        total = 0
        # Pick the largest k values, factoring in the decrement for each turn (i)
        for i in range(k):
            # Compute the effective happiness after i decrements (cannot drop below 0)
            total += max(happiness[i] - i, 0)
        
        return total