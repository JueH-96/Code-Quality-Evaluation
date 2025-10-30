class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the tower constraints in descending order
        maximumHeight.sort(reverse=True)

        total_sum = 0
        # "previous" tracks the previously chosen height (start with a large number)
        prev_chosen = float('inf')

        for max_h in maximumHeight:
            # Choose the maximum possible height that is strictly less than "prev_chosen"
            chosen_height = min(max_h, prev_chosen - 1)

            if chosen_height <= 0:
                # If we cannot assign a positive height, return -1
                return -1

            total_sum += chosen_height
            prev_chosen = chosen_height

        return total_sum