class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        def is_palindrome(s):
            return s == s[::-1]
        
        def count_chars(s):
            char_count = [0] * 26
            for char in s:
                char_count[ord(char) - ord('a')] += 1
            return char_count
        
        def can_form_palindrome(s1, s2):
            char_count1 = count_chars(s1)
            char_count2 = count_chars(s2)
            for i in range(26):
                if char_count1[i] % 2 != char_count2[i] % 2:
                    return False
            return True
        
        n = len(s)
        half = n // 2
        answer = []
        
        for a, b, c, d in queries:
            left = s[a:b+1]
            right = s[c:d+1]
            left_mirror = s[half + a:half + b + 1]
            right_mirror = s[n - d - 1:n - c - 1:-1]
            
            if is_palindrome(left + right) or is_palindrome(left_mirror + right_mirror) or \
               can_form_palindrome(left, right_mirror) or can_form_palindrome(left_mirror, right):
                answer.append(True)
            else:
                answer.append(False)
        
        return answer