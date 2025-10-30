class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        # Early exit: if s contains fewer than k ones in total, return ""
        if s.count('1') < k:
            return ""
        
        # Build a prefix sum array for counting 1's quickly.
        # prefix[i] is the number of 1's in s[:i]
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + (1 if s[i] == '1' else 0)
        
        best_len = float('inf')
        best_str = None
        
        # Iterate over every possible starting index.
        for i in range(n):
            # For each starting index, find the shortest substring s[i:j+1] that has exactly k ones.
            for j in range(i, n):
                # Count of ones in s[i:j+1]:
                if prefix[j+1] - prefix[i] == k:
                    curr_len = j - i + 1
                    candidate = s[i:j+1]
                    # Check if this candidate improves our result.
                    if curr_len < best_len:
                        best_len = curr_len
                        best_str = candidate
                    elif curr_len == best_len and candidate < best_str:
                        best_str = candidate
                    # No need to extend further since adding extra zeros would only increase length.
                    break
        
        return best_str if best_str is not None else ""
        
# Example usage:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    s = "100011001"
    k = 3
    print(sol.shortestBeautifulSubstring(s, k))  # Expected output: "11001"
    
    # Example 2:
    s = "1011"
    k = 2
    print(sol.shortestBeautifulSubstring(s, k))  # Expected output: "11"
    
    # Example 3:
    s = "000"
    k = 1
    print(sol.shortestBeautifulSubstring(s, k))  # Expected output: ""