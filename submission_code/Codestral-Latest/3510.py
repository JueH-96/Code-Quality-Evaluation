class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the maximum heights in ascending order
        maximumHeight.sort()

        # Initialize the total sum
        total_sum = 0

        # Iterate through the sorted maximum heights
        for i in range(len(maximumHeight)):
            # If the current maximum height is less than or equal to the current index,
            # it's not possible to assign unique heights
            if maximumHeight[i] <= i:
                return -1
            # Otherwise, add the current index + 1 to the total sum
            total_sum += i + 1

        return total_sum