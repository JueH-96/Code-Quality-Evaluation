class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_length = 0
        for i in range(len(word)):
            for j in range(i, len(word)):
                substring = word[i:j+1]
                is_valid = True
                for k in range(len(substring)):
                    for l in range(k, len(substring)):
                        if substring[k:l+1] in forbidden_set:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if is_valid:
                    max_length = max(max_length, len(substring))
        return max_length