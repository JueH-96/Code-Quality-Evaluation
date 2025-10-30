class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dp(last_two, x, y, z):
            if x == 0 and y == 0 and z == 0:
                return 0

            max_length = 0

            # Try adding "AA"
            if x > 0:
                if not last_two.endswith("AA"):
                    new_last_two = (last_two + "AA")[-2:]
                    length = 2 + dp(new_last_two, x-1, y, z)
                    if length > max_length:
                        max_length = length

            # Try adding "BB"
            if y > 0:
                if not last_two.endswith("BB"):
                    new_last_two = (last_two + "BB")[-2:]
                    length = 2 + dp(new_last_two, x, y-1, z)
                    if length > max_length:
                        max_length = length

            # Try adding "AB"
            if z > 0:
                if not last_two.endswith("AA"):
                    new_last_two = (last_two + "AB")[-2:]
                    length = 2 + dp(new_last_two, x, y, z-1)
                    if length > max_length:
                        max_length = length

            return max_length

        # Start with an empty string
        return dp("", x, y, z)