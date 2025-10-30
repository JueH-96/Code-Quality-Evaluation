class Solution:
    def isFascinating(self, n: int) -> bool:
        # Concatenate n, 2*n, and 3*n into one string
        concatenated = str(n) + str(n * 2) + str(n * 3)
        # The concatenated string should have exactly 9 characters,
        # and when sorted it should be "123456789"
        return ''.join(sorted(concatenated)) == "123456789"