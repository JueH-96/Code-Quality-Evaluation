class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        answer = []
        
        # Precompute the frequency of each character at each position
        freq = [[0] * 26 for _ in range(n + 1)]
        for i in range(n):
            freq[i + 1] = freq[i].copy()
            freq[i + 1][ord(s[i]) - ord('a')] += 1
        
        for a, b, c, d in queries:
            # Check if the substrings s[a:b+1] and s[c:d+1] can be rearranged to form a palindrome
            left = freq[b + 1] - freq[a]
            right = freq[d + 1] - freq[c]
            
            # Check if the substrings are palindromes
            is_palindrome = True
            for i in range(26):
                if (left[i] % 2 != right[i] % 2):
                    is_palindrome = False
                    break
            
            answer.append(is_palindrome)
        
        return answer