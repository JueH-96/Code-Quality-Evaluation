class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        # Initialize an empty set to keep track of the elements in the array
        elements = set()
        # Initialize the sum of the array
        total_sum = 0
        # Initialize the current number to be added to the array
        current = 1

        while len(elements) < n:
            # Check if the current number or k - current number is in the set
            if current not in elements and k - current not in elements:
                # Add the current number to the set
                elements.add(current)
                # Add the current number to the total sum
                total_sum += current
            # Increment the current number
            current += 1

        return total_sum