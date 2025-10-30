class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        def kmp(pattern):
            m = len(pattern)
            pi = [0] * m
            j = 0
            for i in range(1, m):
                while j > 0 and pattern[i] != pattern[j]:
                    j = pi[j-1]
                if pattern[i] == pattern[j]:
                    j += 1
                pi[i] = j
            return pi

        def find_all_occurrences(text, pattern):
            pi = kmp(pattern)
            occurrences = []
            j = 0
            for i in range(len(text)):
                while j > 0 and text[i] != pattern[j]:
                    j = pi[j-1]
                if text[i] == pattern[j]:
                    j += 1
                if j == len(pattern):
                    occurrences.append(i - len(pattern) + 1)
                    j = pi[j-1]
            return occurrences

        a_indices = find_all_occurrences(s, a)
        b_indices = find_all_occurrences(s, b)

        beautiful_indices = []
        j = 0
        for i in a_indices:
            while j < len(b_indices) and b_indices[j] < i - k:
                j += 1
            if j < len(b_indices) and abs(b_indices[j] - i) <= k:
                beautiful_indices.append(i)

        return beautiful_indices