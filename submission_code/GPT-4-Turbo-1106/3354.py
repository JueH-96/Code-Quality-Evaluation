class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # Initialize an array to keep track of the count of each character
        char_count = [0] * 26
        
        # Convert the string to a list for easier manipulation
        s_list = list(s)
        
        # Iterate through the string and replace '?' with the lexicographically smallest character
        for i in range(len(s_list)):
            if s_list[i] == '?':
                # Find the lexicographically smallest character that minimizes the value
                for j in range(26):
                    if char_count[j] == 0:
                        s_list[i] = chr(97 + j)
                        break
            # Update the character count
            char_count[ord(s_list[i]) - 97] += 1
        
        # Convert the list back to a string and return
        return ''.join(s_list)

# Example usage:
# sol = Solution()
# print(sol.minimizeStringValue("??"))  # Output: "ab"
# print(sol.minimizeStringValue("a?a?"))  # Output: "abac"