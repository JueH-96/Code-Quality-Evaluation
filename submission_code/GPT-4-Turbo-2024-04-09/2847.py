class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        reversed_map = {}
        max_pairs = 0
        
        for word in words:
            reversed_word = word[::-1]
            if reversed_word in reversed_map and reversed_map[reversed_word] > 0:
                max_pairs += 1
                reversed_map[reversed_word] -= 1
            else:
                if word not in reversed_map:
                    reversed_map[word] = 0
                reversed_map[word] += 1
        
        return max_pairs