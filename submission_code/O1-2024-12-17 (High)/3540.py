class Solution:
    def stringHash(self, s: str, k: int) -> str:
        n = len(s)
        result = []
        
        for i in range(0, n, k):
            substring = s[i:i+k]
            total = 0
            for ch in substring:
                # Convert character to its hash value (a->0, b->1, ..., z->25)
                total += (ord(ch) - ord('a'))
            # Compute remainder when divided by 26
            hashed_value = total % 26
            # Convert back to character and append to result
            result.append(chr(hashed_value + ord('a')))
        
        return "".join(result)