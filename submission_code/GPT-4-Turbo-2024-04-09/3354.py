class Solution:
    def minimizeStringValue(self, s: str) -> str:
        from collections import defaultdict
        
        # Convert the string to a list of characters for mutability
        s = list(s)
        n = len(s)
        
        # This will hold the count of each character seen so far
        char_count = defaultdict(int)
        
        # Iterate over each character in the string
        for i in range(n):
            if s[i] == '?':
                # Find the lexicographically smallest character that minimizes the cost
                for ch in 'abcdefghijklmnopqrstuvwxyz':
                    if char_count[ch] == 0:
                        s[i] = ch
                        char_count[ch] += 1
                        break
            else:
                # Increment the count of the current character
                char_count[s[i]] += 1
        
        # Join the list back into a string
        return ''.join(s)