class Solution:
    def getGoodIndices(self, variables, target):
        good_indices = []
        for i in range(len(variables)):
            if (((variables[i][0] ** variables[i][1]) % 10) ** variables[i][2]) % variables[i][3] == target:
                good_indices.append(i)
        return good_indices