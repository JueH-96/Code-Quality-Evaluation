class Solution:
    def punishmentNumber(self, n: int) -> int:
        punishment_sum = 0

        # Helper function for backtracking
        # s_str: string representation of i * i
        # current_index: current position in s_str
        # current_sum: sum of partitioned substrings so far
        # target: the integer i we are trying to match
        def can_partition_sum(s_str, current_index, current_sum, target):
            # Base case 1: Reached the end of the string
            # If we have processed the entire string, check if the sum matches the target
            if current_index == len(s_str):
                return current_sum == target

            # Recursive step: Try all possible partition points from the current index
            # The substring goes from current_index to j (inclusive)
            for j in range(current_index, len(s_str)):
                # Extract the substring from current_index to j + 1
                substring = s_str[current_index : j + 1]

                # Convert substring to integer value
                # This should always be valid since s_str comes from a number string containing digits
                val = int(substring)

                # Calculate the sum if we take this substring
                next_sum = current_sum + val

                # Pruning Check: If adding this value already exceeds the target
                # Since 'val' will not decrease for longer substrings starting
                # at current_index and are generally positive (except for '0'), no further substrings
                # starting at current_index will result in a sum <= target.
                # We can safely stop exploring partitions that start at current_index
                # and include a substring ending at or after j.
                if next_sum > target:
                    break # Exit the inner loop (loop over j)

                # If adding this value does not exceed the target, explore this path
                # Recursively call for the rest of the string starting from index j + 1
                # Only proceed if the recursive call finds a valid path
                if can_partition_sum(s_str, j + 1, next_sum, target):
                    return True # Found a valid partition path

            # If the loop finishes without finding a valid partition starting from current_index
            # that leads to the target sum
            return False


        # Iterate through integers from 1 to n
        for i in range(1, n + 1):
            square = i * i
            square_str = str(square)

            # Check if the square's string representation can be partitioned
            # Initial call starts from index 0, with a current_sum of 0, aiming for target i
            if can_partition_sum(square_str, 0, 0, i):
                punishment_sum += square

        return punishment_sum