class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        result = []
        
        for remove_idx in range(n):
            if n - 1 < k:  # After removing one word, less than k remain
                result.append(0)
                continue
            
            # Find longest common prefix among any k strings (excluding remove_idx)
            max_prefix_len = 0
            
            # Generate all possible prefixes and count occurrences
            prefix_count = {}
            for i in range(n):
                if i == remove_idx:
                    continue
                word = words[i]
                for j in range(1, len(word) + 1):
                    prefix = word[:j]
                    prefix_count[prefix] = prefix_count.get(prefix, 0) + 1
            
            # Find the longest prefix that appears at least k times
            for prefix, count in prefix_count.items():
                if count >= k:
                    max_prefix_len = max(max_prefix_len, len(prefix))
            
            result.append(max_prefix_len)
        
        return result