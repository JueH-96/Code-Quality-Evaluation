class Solution:
    def minimizeConcatenatedLength(self, words: list[str]) -> int:
        def join(x, y):
            if x[-1] == y[0]:
                return x + y[1:]
            else:
                return x + y
        
        # Initialize the concatenated string with the first word
        concatenated = words[0]
        
        # Iterate over the words to perform the join operations
        for i in range(1, len(words)):
            # Check both possible join operations and choose the one with minimum length
            join1 = join(concatenated, words[i])
            join2 = join(words[i], concatenated)
            concatenated = min(join1, join2, key=len)
        
        return len(concatenated)

# Example usage:
# sol = Solution()
# print(sol.minimizeConcatenatedLength(["aa","ab","bc"])) # Output: 4
# print(sol.minimizeConcatenatedLength(["ab","b"])) # Output: 2
# print(sol.minimizeConcatenatedLength(["aaa","c","aba"])) # Output: 6