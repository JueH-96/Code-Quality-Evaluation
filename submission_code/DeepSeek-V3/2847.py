class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        used = set()
        count = 0
        for i in range(len(words)):
            if i in used:
                continue
            for j in range(i+1, len(words)):
                if j in used:
                    continue
                if words[i] == words[j][::-1]:
                    used.add(i)
                    used.add(j)
                    count += 1
                    break
        return count