class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        total_length = n + m - 1
        
        # Initialize the result array with a character that is lexicographically small and not in str2
        result = ['a'] * total_length
        
        # Helper function to check if a substring matches str2
        def matches_at(word, index):
            return word[index:index + m] == list(str2)
        
        # Helper function to generate a non-matching substring
        def non_matching_substring():
            # Start with the same characters as str2 and change the first character
            if str2[0] == 'a':
                return ['b'] + list(str2[1:])
            else:
                return ['a'] + list(str2[1:])
        
        # Apply the rules from str1
        for i in range(n):
            if str1[i] == 'T':
                # Place str2 starting at index i in result
                result[i:i + m] = list(str2)
            elif str1[i] == 'F':
                # Place a non-matching substring starting at index i
                non_match = non_matching_substring()
                if matches_at(result, i):
                    # If it accidentally matches, we need to ensure it doesn't
                    result[i:i + m] = non_match
        
        # Verify the result satisfies all conditions
        for i in range(n):
            if str1[i] == 'T' and not matches_at(result, i):
                return ""  # Failed to match where it should
            if str1[i] == 'F' and matches_at(result, i):
                return ""  # Matched where it should not
        
        return ''.join(result)

# Example usage:
sol = Solution()
print(sol.generateString("TFTF", "ab"))  # Output: "ababa"
print(sol.generateString("TFTF", "abc")) # Output: ""
print(sol.generateString("F", "d"))      # Output: "a"