class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        result = ""
        
        for i in range(n):
            count_ones = 0
            for j in range(i, n):
                if s[j] == '1':
                    count_ones += 1
                if count_ones == k:
                    current_substring = s[i:j+1]
                    if len(current_substring) < min_len or (len(current_substring) == min_len and current_substring < result):
                        min_len = len(current_substring)
                        result = current_substring
                    break
        
        return result

# Example usage:
# sol = Solution()
# print(sol.shortestBeautifulSubstring("100011001", 3))  # Output: "11001"
# print(sol.shortestBeautifulSubstring("1011", 2))       # Output: "11"
# print(sol.shortestBeautifulSubstring("000", 1))        # Output: ""