class Solution:
    def stringHash(self, s: str, k: int) -> str:
        n = len(s)
        num_substrings = n // k
        result = ""

        for i in range(num_substrings):
            substring = s[i * k : (i + 1) * k]
            hash_sum = 0
            for char in substring:
                hash_value = ord(char) - ord('a')
                hash_sum += hash_value

            hashedChar_value = hash_sum % 26
            hashed_char = chr(ord('a') + hashedChar_value)
            result += hashed_char

        return result