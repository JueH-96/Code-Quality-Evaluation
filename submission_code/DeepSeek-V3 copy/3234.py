class Solution:
    def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
        good_indices = []
        for i, (a, b, c, m) in enumerate(variables):
            # Compute a^b % 10
            first_step = pow(a, b, 10)
            # Compute (first_step)^c % m
            second_step = pow(first_step, c, m)
            if second_step == target:
                good_indices.append(i)
        return good_indices