class Solution:
    def punishmentNumber(self, n: int) -> int:
        # Helper function that checks if we can partition string s into segments that sum to target.
        def can_partition(s: str, target: int, start: int) -> bool:
            # Base: if we have reached the end, check if we've exactly hit target.
            if start == len(s):
                return target == 0
            # Try every possible partition starting at index "start"
            current_val = 0
            for i in range(start, len(s)):
                # Building the substring number on the fly.
                current_val = current_val * 10 + int(s[i])
                # Only proceed if current_val doesn't exceed target.
                if current_val > target:
                    continue  # We can actually break here if s has no leading zeros.
                if can_partition(s, target - current_val, i + 1):
                    return True
            return False

        punishment_sum = 0
        for i in range(1, n + 1):
            square_str = str(i * i)
            if can_partition(square_str, i, 0):
                punishment_sum += i * i
        return punishment_sum