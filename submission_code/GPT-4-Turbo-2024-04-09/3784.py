class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        from collections import Counter
        
        def common_prefix_length(str1, str2):
            min_len = min(len(str1), len(str2))
            for i in range(min_len):
                if str1[i] != str2[i]:
                    return i
            return min_len
        
        n = len(words)
        if k > n:
            return [0] * n
        
        answer = [0] * n
        
        for i in range(n):
            # Create a new list excluding the i-th element
            new_words = words[:i] + words[i+1:]
            
            # Count occurrences of each word
            word_count = Counter(new_words)
            
            # Find the word with the maximum count (at least k occurrences)
            max_prefix = 0
            for word, count in word_count.items():
                if count >= k:
                    # If the count is at least k, we can consider the word itself
                    max_prefix = max(max_prefix, len(word))
                else:
                    # Otherwise, we need to consider combinations
                    # Since this is computationally expensive, we use a simpler approach:
                    # We find the longest common prefix of this word with others until we reach k
                    current_prefix_length = len(word)
                    occurrences = 1  # Start with the current word itself
                    for other_word in new_words:
                        if other_word == word:
                            continue
                        current_prefix_length = min(current_prefix_length, common_prefix_length(word, other_word))
                        occurrences += 1
                        if occurrences == k:
                            break
                    max_prefix = max(max_prefix, current_prefix_length)
            
            answer[i] = max_prefix
        
        return answer