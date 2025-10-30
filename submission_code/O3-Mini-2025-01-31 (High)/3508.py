class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # For a successful conversion, every bit that is 1 in k must also be 1 in n.
        # If (n & k) != k, it means there is a bit set in k that is not set in n,
        # and since we can only change 1s to 0s, it's impossible to form k.
        if (n & k) != k:
            return -1

        # Count how many 1s are in n and k.
        # We are allowed only to turn extra 1s in n into 0s.
        change_count = bin(n).count("1") - bin(k).count("1")
        
        return change_count