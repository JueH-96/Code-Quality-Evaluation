class Solution:
    def kthCharacter(self, k: int) -> str:
        # initial string
        word = "a"
        # while word length is less than k keep performing the operations
        while len(word) < k:
            # generate new string by shifting each character in word by 1 in the alphabet
            new_str = ''.join(chr(((ord(c) - ord('a') + 1) % 26) + ord('a')) for c in word)
            word += new_str
        # Return the kth character (1-indexed)
        return word[k-1]
        
# Sample Test Cases
if __name__ == "__main__":
    sol = Solution()
    print(sol.kthCharacter(5))  # Expected output: "b"
    print(sol.kthCharacter(10)) # Expected output: "c"