class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        n = len(words)
        used = [False] * n
        count = 0
        for i in range(n):
            if not used[i]:
                reversed_word = words[i][::-1]
                for j in range(i + 1, n):
                    if not used[j] and words[j] == reversed_word:
                        used[i] = True
                        used[j] = True
                        count += 1
                        break
        return count