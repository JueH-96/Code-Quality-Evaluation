class Solution:
    def isFascinating(self, n: int) -> bool:
        # Concatenate n, 2*n, and 3*n into one string
        s = f"{n}{2*n}{3*n}"
        # Check if the concatenated string has length 9 and contains all digits 1-9 exactly once
        return len(s) == 9 and set(s) == set("123456789")