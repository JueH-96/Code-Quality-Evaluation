class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness values in descending order
        happiness.sort(reverse=True)

        # Initialize the total happiness sum
        total_happiness = 0

        # Iterate through the first k elements
        for i in range(k):
            # Add the current happiness value to the total happiness sum
            total_happiness += happiness[i]

            # Decrease the happiness values of the remaining children
            for j in range(i + 1, len(happiness)):
                if happiness[j] > 0:
                    happiness[j] -= 1

        return total_happiness