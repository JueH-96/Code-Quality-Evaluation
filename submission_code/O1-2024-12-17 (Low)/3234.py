class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for i, (a, b, c, m) in enumerate(variables):
            # First compute (a^b) mod 10
            x = pow(a, b, 10)
            # Then compute (x^c) mod m
            result = pow(x, c, m)
            # Check if it matches target
            if result == target:
                good_indices.append(i)
        return good_indices