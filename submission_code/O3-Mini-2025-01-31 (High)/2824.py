class Solution:
    def isFascinating(self, n: int) -> bool:
        # Concatenate n, 2*n, and 3*n into a single string.
        concatenated = str(n) + str(n * 2) + str(n * 3)
        # Check if the concatenated string is exactly 9 digits long
        # and contains all digits from '1' to '9' exactly once.
        return len(concatenated) == 9 and set(concatenated) == set("123456789")