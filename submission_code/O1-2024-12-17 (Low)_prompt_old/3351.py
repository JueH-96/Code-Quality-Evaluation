class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness values in descending order
        happiness.sort(reverse=True)
        
        # We will pick the top k happiness values in this sorted list,
        # choosing the largest child first (turn 0), then the second largest (turn 1), etc.
        # The final happiness of the j-th chosen child will be max(0, happiness[j] - j).
        
        total = 0
        for j in range(k):
            total += max(0, happiness[j] - j)
        
        return total