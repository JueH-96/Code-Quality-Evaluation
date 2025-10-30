from typing import List

class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for i, (a, b, c, m) in enumerate(variables):
            # compute (a^b % 10)^c % m
            # First, compute a^b mod 10
            intermediate = pow(a, b, 10)
            # Then compute (intermediate)^c mod m
            result = pow(intermediate, c, m)
            if result == target:
                good_indices.append(i)
        return good_indices

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    variables = [[2, 3, 3, 10], [3, 3, 3, 1], [6, 1, 1, 4]]
    target = 2
    print(sol.getGoodIndices(variables, target))  # Expected output: [0, 2]

    # Example 2
    variables = [[39, 3, 1000, 1000]]
    target = 17
    print(sol.getGoodIndices(variables, target))  # Expected output: []