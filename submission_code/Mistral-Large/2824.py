class Solution:
    def isFascinating(self, n: int) -> bool:
        # Concatenate n, 2*n, and 3*n
        concatenated = str(n) + str(2 * n) + str(3 * n)

        # Check if the concatenated string contains all digits from 1 to 9 exactly once
        return ''.join(sorted(concatenated)) == '123456789'