class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        count = 0
        seen = set()
        for i in range(len(words)):
            if words[i] in seen:
                continue
            for j in range(i + 1, len(words)):
                if words[i] == words[j][::-1]:
                    count += 1
                    seen.add(words[i])
                    seen.add(words[j])
                    break
        return count