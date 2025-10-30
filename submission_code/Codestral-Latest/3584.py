class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n, m = len(word1), len(word2)
        if m > n:
            return []

        # Helper function to check if two strings are almost equal
        def almost_equal(s1, s2):
            diff_count = 0
            for c1, c2 in zip(s1, s2):
                if c1 != c2:
                    diff_count += 1
                if diff_count > 1:
                    return False
            return True

        # Try to find the lexicographically smallest valid sequence of indices
        result = []
        i, j = 0, 0
        while i < n and j < m:
            if word1[i] == word2[j]:
                result.append(i)
                i += 1
                j += 1
            else:
                i += 1

        # Check if the remaining part of word1 is almost equal to word2[j:]
        if j < m and almost_equal(word1[i:], word2[j:]):
            result.extend(range(i, n))

        # If the result length matches word2 length, return the result
        if len(result) == m:
            return result

        # Otherwise, return an empty array
        return []