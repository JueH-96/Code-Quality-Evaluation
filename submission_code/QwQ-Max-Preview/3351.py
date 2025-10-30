class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness array in descending order to maximize the sum
        happiness.sort(reverse=True)
        total = 0
        # Iterate through the first k elements
        for i in range(k):
            current = happiness[i] - i
            # Add only positive values to the total sum
            if current > 0:
                total += current
        return total