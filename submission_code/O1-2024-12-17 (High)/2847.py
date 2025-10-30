class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        used = set()
        count = 0
        n = len(words)
        
        for i in range(n):
            if i in used:
                continue
            for j in range(i + 1, n):
                if j in used:
                    continue
                # Check if words[i] and words[j] are reversed pairs
                if words[i] == words[j][::-1]:
                    count += 1
                    used.add(i)
                    used.add(j)
                    break
        
        return count