class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def is_valid(sub):
            # Check if each character appears exactly k times
            for char in set(sub):
                if sub.count(char) != k:
                    return False
            # Check if the difference between adjacent characters is at most 2
            for i in range(len(sub) - 1):
                if abs(ord(sub[i]) - ord(sub[i + 1])) > 2:
                    return False
            return True
        
        count = 0
        # Generate all possible substrings of length k * unique characters in the substring
        for i in range(len(word)):
            for j in range(i + k, min(i + k * 26, len(word)) + 1, k):
                if is_valid(word[i:j]):
                    count += 1
        return count

# Example usage:
# sol = Solution()
# print(sol.countCompleteSubstrings("igigee", 2))  # Output: 3
# print(sol.countCompleteSubstrings("aaabbbccc", 3))  # Output: 6