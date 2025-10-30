class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        n = len(word)
        forbidden_set = set(forbidden)
        max_len = 0
        for i in range(n):
            for j in range(i, n):
                substring = word[i:j+1]
                is_valid = True
                for k in range(len(substring)):
                    for l in range(k, len(substring)):
                        sub_substring = substring[k:l+1]
                        if sub_substring in forbidden_set:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if is_valid:
                    max_len = max(max_len, len(substring))
        return max_len