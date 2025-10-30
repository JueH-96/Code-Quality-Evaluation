class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        ans = 0
        for i in range(n):
            counts = {}
            counts[word[i]] = 1
            last_char = word[i]
            valid = True
            if counts[word[i]] > k:
                continue
            letters_in_substring = set([word[i]])
            if counts[word[i]] == k and all(counts[c] == k for c in letters_in_substring):
                ans += 1
            for j in range(i+1, n):
                if abs(ord(word[j]) - ord(last_char)) > 2:
                    break
                last_char = word[j]
                counts[word[j]] = counts.get(word[j], 0) + 1
                letters_in_substring.add(word[j])
                if counts[word[j]] > k:
                    break
                if all(counts[c] == k for c in letters_in_substring):
                    ans += 1
        return ans