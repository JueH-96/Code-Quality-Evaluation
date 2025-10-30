class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        count = 0
        used = set()
        for i in range(len(words)):
            if i in used:
                continue
            reversed_word = words[i][::-1]
            for j in range(i + 1, len(words)):
                if j in used:
                    continue
                if words[j] == reversed_word:
                    count += 1
                    used.add(i)
                    used.add(j)
                    break
        return count