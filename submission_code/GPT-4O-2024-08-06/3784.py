from typing import List
from collections import defaultdict

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        def common_prefix_length(s1, s2):
            """Helper function to calculate the length of the common prefix of two strings."""
            min_length = min(len(s1), len(s2))
            for i in range(min_length):
                if s1[i] != s2[i]:
                    return i
            return min_length
        
        n = len(words)
        answer = [0] * n
        
        # Create a dictionary to count occurrences of each word
        word_count = defaultdict(int)
        for word in words:
            word_count[word] += 1
        
        for i in range(n):
            # Remove the current word from the count
            current_word = words[i]
            word_count[current_word] -= 1
            
            # If there are fewer than k words left, answer is 0
            if sum(word_count.values()) < k:
                answer[i] = 0
            else:
                # Find the longest common prefix among any k words
                max_prefix_length = 0
                for word, count in word_count.items():
                    if count >= k:
                        # If there are at least k occurrences of this word, the prefix is the word itself
                        max_prefix_length = max(max_prefix_length, len(word))
                    elif count > 0:
                        # Otherwise, find the longest common prefix with other words
                        for other_word, other_count in word_count.items():
                            if other_word != word and other_count > 0:
                                prefix_length = common_prefix_length(word, other_word)
                                max_prefix_length = max(max_prefix_length, prefix_length)
                
                answer[i] = max_prefix_length
            
            # Restore the current word count
            word_count[current_word] += 1
        
        return answer