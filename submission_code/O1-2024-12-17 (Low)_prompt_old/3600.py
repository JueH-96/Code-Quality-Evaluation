class Solution:
    def kthCharacter(self, k: int) -> str:
        def next_string(s: str) -> str:
            # Transform each character to the next character
            # 'z' wraps around to 'a'.
            result_chars = []
            for c in s:
                if c == 'z':
                    result_chars.append('a')
                else:
                    result_chars.append(chr(ord(c) + 1))
            return ''.join(result_chars)

        word = "a"
        # Expand word until its length is at least k
        while len(word) < k:
            word += next_string(word)

        return word[k - 1]