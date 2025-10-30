class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        memo = {}

        def solve(index, first, last):
            if index == n:
                return 0
            
            if (index, first, last) in memo:
                return memo[(index, first, last)]

            word = words[index]
            res = float('inf')

            # Option 1: Join current word to the right
            join_len1 = len(word)
            if last == word[0]:
                join_len1 -= 1
            res = min(res, join_len1 + solve(index + 1, first, word[-1]))

            # Option 2: Join current word to the left
            join_len2 = len(word)
            if first == word[-1]:
                join_len2 -=1
            res = min(res, join_len2 + solve(index + 1, word[0], last))
            
            memo[(index, first, last)] = res
            return res

        first_char = words[0][0]
        last_char = words[0][-1]
        ans = len(words[0]) + solve(1, first_char, last_char)
        return ans