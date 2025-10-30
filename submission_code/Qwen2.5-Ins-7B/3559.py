class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        from collections import Counter
        
        # Count the frequency of each prefix in words
        prefix_count = Counter()
        for word in words:
            for i in range(1, len(word) + 1):
                prefix_count[word[:i]] += 1
        
        # Initialize the result and the index in target
        result = 0
        i = 0
        
        while i < len(target):
            found = False
            for j in range(min(len(target) - i, 5000), 0, -1):
                if target[i:i+j] in prefix_count:
                    result += 1
                    i += j
                    found = True
                    break
            if not found:
                return -1
        
        return result