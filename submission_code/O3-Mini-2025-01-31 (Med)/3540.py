class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = []
        # Process the string in chunks of k characters.
        for i in range(0, len(s), k):
            substring = s[i:i+k]
            total_sum = sum(ord(c) - ord('a') for c in substring)  # Calculate the sum of character indices.
            hashed_value = total_sum % 26  # Compute modulo 26.
            result.append(chr(hashed_value + ord('a')))  # Map back to a character.
        return ''.join(result)