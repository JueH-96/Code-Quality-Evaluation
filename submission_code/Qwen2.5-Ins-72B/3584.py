from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n, m = len(word1), len(word2)
        dp = [-1] * m
        char_indices = {char: [] for char in set(word2)}
        
        for i, char in enumerate(word1):
            if char in char_indices:
                char_indices[char].append(i)
        
        def find_next_index(arr, target):
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] > target:
                    right = mid - 1
                else:
                    left = mid + 1
            return left if left < len(arr) else -1
        
        for i in range(m):
            if i == 0:
                for char in word2[i] + chr(ord(word2[i]) + 1):
                    if char in char_indices:
                        next_index = find_next_index(char_indices[char], -1)
                        if next_index != -1:
                            dp[i] = char_indices[char][next_index]
                            break
            else:
                for char in word2[i] + chr(ord(word2[i]) + 1):
                    if char in char_indices:
                        next_index = find_next_index(char_indices[char], dp[i-1])
                        if next_index != -1:
                            dp[i] = char_indices[char][next_index]
                            break
            if dp[i] == -1:
                return []
        
        return dp