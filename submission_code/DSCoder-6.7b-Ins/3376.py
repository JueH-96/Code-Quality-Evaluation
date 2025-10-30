class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Create a reverse index for wordsContainer
        rev_index = {w[::-1]: i for i, w in enumerate(wordsContainer)}
        # Initialize an empty list to store the result
        result = []
        # Iterate over wordsQuery
        for q in wordsQuery:
            # Find the longest common suffix of q in rev_index
            for i in range(len(q), 0, -1):
                if q[-i:] in rev_index:
                    # Append the index of the string in wordsContainer to result
                    result.append(rev_index[q[-i:]])
                    break
        # Return the result
        return result