class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        def findLongestCommonPrefix(word_list, k):
            if len(word_list) < k:
                return 0
            
            # Try all possible prefix lengths from longest to shortest
            max_len = max(len(word) for word in word_list) if word_list else 0
            
            for prefix_len in range(max_len, -1, -1):
                # Count prefixes of this length
                prefix_count = {}
                for word in word_list:
                    if len(word) >= prefix_len:
                        prefix = word[:prefix_len]
                        prefix_count[prefix] = prefix_count.get(prefix, 0) + 1
                
                # Check if any prefix appears at least k times
                for count in prefix_count.values():
                    if count >= k:
                        return prefix_len
            
            return 0
        
        result = []
        n = len(words)
        
        for i in range(n):
            # Create remaining words after removing index i
            remaining = words[:i] + words[i+1:]
            # Find longest common prefix among k strings
            result.append(findLongestCommonPrefix(remaining, k))
        
        return result