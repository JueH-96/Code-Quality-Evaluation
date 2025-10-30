class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        # Sort the capacity array in descending order
        capacity.sort(reverse=True)

        # Initialize variables to keep track of the total capacity used and the number of boxes used
        total_capacity = 0
        boxes_used = 0

        # Iterate through the apple array
        for apples in apple:
            # If the current box cannot hold the apples, move to the next box
            if total_capacity + apples > capacity[boxes_used]:
                boxes_used += 1
                total_capacity = 0

            # Add the apples to the current box
            total_capacity += apples

        # Return the number of boxes used
        return boxes_used + 1