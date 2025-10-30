class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        last_invalid_index = -1
        max_length = 0
        n = len(word)
        max_forbidden_length = max(len(fw) for fw in forbidden)
        
        for i in range(n):
            for l in range(1, min(max_forbidden_length, i + 1) + 1):
                s = word[i - l + 1:i +1]
                if s in forbidden_set:
                    last_invalid_index = max(last_invalid_index, i - l +1)
                    break
            curr_length = i - last_invalid_index
            max_length = max(max_length, curr_length)
        return max_length