class Solution:
    def punishmentNumber(self, n: int) -> int:
        # Helper function to check if we can partition s (the stringified square)
        # into contiguous substrings that sum up to target.
        def can_partition(s, target, index, current_sum):
            # Base: if we've used all characters, check if the sum equals the target.
            if index == len(s):
                return current_sum == target
            # Prune paths that already exceed the target.
            if current_sum > target:
                return False
            # Try every possible split from current index.
            for j in range(index + 1, len(s) + 1):
                # Convert the current partitioned substring to an integer.
                num = int(s[index:j])
                if can_partition(s, target, j, current_sum + num):
                    return True
            return False

        total = 0
        # Check for every i from 1 to n.
        for i in range(1, n + 1):
            square = i * i
            s = str(square)
            # If i^2 can be partitioned so that the sum equals i, add i^2 to total.
            if can_partition(s, i, 0, 0):
                total += square
        return total