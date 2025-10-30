class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        @cache
        def dp(i, first, last):
            if i == len(words):
                return 0
            
            curr_first, curr_last = words[i][0], words[i][-1]
            curr_len = len(words[i])
            
            # Join current word to the end
            option1 = dp(i + 1, first, curr_last) + curr_len - (last == curr_first)
            
            # Join current word to the beginning
            option2 = dp(i + 1, curr_first, last) + curr_len - (curr_last == first)
            
            return min(option1, option2)
        
        return len(words[0]) + dp(1, words[0][0], words[0][-1])