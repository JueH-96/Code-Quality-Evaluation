class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        def is_palindrome(sub):
            return sub == sub[::-1]

        n = len(s)
        half = n // 2
        answer = []

        for a, b, c, d in queries:
            left_sub = s[a:b+1]
            right_sub = s[c:d+1]

            # Check if the substrings can be rearranged to form a palindrome
            if is_palindrome(left_sub) and is_palindrome(right_sub):
                # Check if the remaining parts of the string form a palindrome
                remaining_left = s[:a] + s[b+1:half]
                remaining_right = s[half:c] + s[d+1:]
                if is_palindrome(remaining_left) and is_palindrome(remaining_right):
                    answer.append(True)
                else:
                    answer.append(False)
            else:
                answer.append(False)

        return answer