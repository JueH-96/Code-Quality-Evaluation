class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_length = 0
        n = len(s)
        
        # Iterate over each possible starting point of the substring
        for i in range(n):
            seen_pair = False
            current_length = 1
            
            # Explore the substring starting from index i
            for j in range(i + 1, n):
                # Check if we encounter a second pair of consecutive same digits
                if s[j] == s[j - 1]:
                    if seen_pair:
                        break
                    else:
                        seen_pair = True
                current_length += 1
            
            # Update the maximum length found
            max_length = max(max_length, current_length)
        
        return max_length