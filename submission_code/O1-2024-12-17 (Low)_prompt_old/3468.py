class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        # Since the string is treated in a cyclic manner, we only need k mod n
        k = k % n  
        
        # Build the encrypted string by taking the (i + k)th character mod n for each index i
        encrypted = []
        for i in range(n):
            encrypted.append(s[(i + k) % n])
        
        return "".join(encrypted)