class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for i, (a, b, c, m) in enumerate(variables):
            # Compute the last digit of a^b
            last_digit = pow(a, b, 10)
            # Raise that to the power of c, and take modulo m
            val = pow(last_digit, c, m)
            if val == target:
                good_indices.append(i)
        return good_indices