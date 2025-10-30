class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        @cache
        def dp(idx, first, last):
            if idx == len(words):
                return 0
                
            curr = words[idx]
            curr_first = curr[0]
            curr_last = curr[-1]
            
            # Try appending current word to end
            op1 = dp(idx + 1, first, curr_last) + len(curr)
            if last == curr_first:
                op1 -= 1
                
            # Try prepending current word
            op2 = dp(idx + 1, curr_first, last) + len(curr)
            if curr_last == first:
                op2 -= 1
                
            return min(op1, op2)
            
        first_word = words[0]
        return len(first_word) + dp(1, first_word[0], first_word[-1])