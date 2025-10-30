class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness array in descending order
        happiness.sort(reverse=True)
        total = 0
        for i in range(k):
            # Calculate the current happiness value after decrementing by i
            current_happiness = max(happiness[i] - i, 0)
            total += current_happiness
        return total