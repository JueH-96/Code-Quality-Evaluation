class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half_n = n // 2
        left = [0] * 26
        right = [0] * 26
        result = []
        
        for i in range(half_n):
            left[ord(s[i]) - ord('a')] += 1
        
        for a, b, c, d in queries:
            temp_left = left[:]
            temp_right = right[:]
            
            for i in range(a, b + 1):
                temp_left[ord(s[i]) - ord('a')] -= 1
            
            for i in range(c, d + 1):
                temp_right[ord(s[n - 1 - i]) - ord('a')] -= 1
            
            for i in range(26):
                if temp_left[i] != temp_right[i]:
                    result.append(False)
                    break
            else:
                result.append(True)
        
        return result