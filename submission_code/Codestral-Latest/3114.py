class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0

        # Function to calculate the sum of heights for a given peak
        def calculate_sum(peak):
            sum_heights = maxHeights[peak]
            current_height = maxHeights[peak]

            # Calculate sum for the left side of the peak
            for i in range(peak - 1, -1, -1):
                current_height = min(current_height, maxHeights[i])
                sum_heights += current_height

            current_height = maxHeights[peak]

            # Calculate sum for the right side of the peak
            for i in range(peak + 1, n):
                current_height = min(current_height, maxHeights[i])
                sum_heights += current_height

            return sum_heights

        # Iterate through each possible peak and calculate the sum of heights
        for i in range(n):
            max_sum = max(max_sum, calculate_sum(i))

        return max_sum