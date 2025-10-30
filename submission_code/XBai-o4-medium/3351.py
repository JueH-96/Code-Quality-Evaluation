class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness array in descending order
        happiness.sort(reverse=True)
        total = 0
        for i in range(k):
            # Calculate the adjusted happiness value, ensuring it's not negative
            current = max(happiness[i] - i, 0)
            total += current
        return total