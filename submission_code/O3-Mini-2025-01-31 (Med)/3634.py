class Solution:
    def calculateScore(self, s: str) -> int:
        # Helper function to compute the mirror character.
        # For any character 'c', its mirror is defined as:
        # mirror(c) = chr(ord('z') - (ord(c) - ord('a')))
        def mirror_char(c: str) -> str:
            return chr(ord('z') - (ord(c) - ord('a')))
        
        # Dictionary to keep track of unmarked indices for each letter.
        # For a letter, we store a list of indices where it has been seen and is still unmarked.
        # We'll use the list as a stack so that the last index (largest j) is used, which
        # corresponds to the "closest" index.
        available = {}
        # Initialize for all lower case letters.
        for ch in "abcdefghijklmnopqrstuvwxyz":
            available[ch] = []
        
        total_score = 0
        
        # Process the string left to right.
        for i, ch in enumerate(s):
            # Determine the mirror character for the current letter.
            m = mirror_char(ch)
            
            # Check if there exists an unmarked index j with the mirror character.
            if available[m]:
                # The latest index in the stack is the closest valid index j.
                j = available[m].pop()
                total_score += i - j
            else:
                # Otherwise, push the current index into the stack for this character.
                available[ch].append(i)
        
        return total_score

# Below is an example showing how the class can be used:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    s1 = "aczzx"
    print(sol.calculateScore(s1))  # Expected output: 5
    
    # Example 2
    s2 = "abcdef"
    print(sol.calculateScore(s2))  # Expected output: 0