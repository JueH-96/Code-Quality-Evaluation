class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        ans = 0
        for length in range(1, 27):
            size = length * k
            if size > n:
                break
            count = [0] * 26
            satisfy = 0
            for i in range(n):
                if i >= size:
                    count[ord(word[i - size]) - ord('a')] -= 1
                    if count[ord(word[i - size]) - ord('a')] == k - 1:
                        satisfy -= 1
                    elif count[ord(word[i - size]) - ord('a')] == k:
                        satisfy += 1
                count[ord(word[i]) - ord('a')] += 1
                if count[ord(word[i]) - ord('a')] == k + 1:
                    satisfy -= 1
                elif count[ord(word[i]) - ord('a')] == k:
                    satisfy += 1
                if i >= size - 1:
                    valid = True
                    for j in range(i - size + 2, i + 1):
                        if abs(ord(word[j]) - ord(word[j - 1])) > 2:
                            valid = False
                            break
                    if valid and satisfy == length:
                        ans += 1
        return ans