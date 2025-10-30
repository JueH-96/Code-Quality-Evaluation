class Solution:
    def compressedString(self, word: str) -> str:
        comp = []
        i = 0
        n = len(word)
        while i < n:
            c = word[i]
            j = i
            while j < n and word[j] == c:
                j += 1
            run_length = j - i
            while run_length > 0:
                chunk = min(9, run_length)
                comp.append(f"{chunk}{c}")
                run_length -= chunk
                i += chunk
        return ''.join(comp)