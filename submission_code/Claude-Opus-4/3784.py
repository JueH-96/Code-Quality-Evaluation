class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        from collections import defaultdict
        
        n = len(words)
        result = []
        
        for i in range(n):
            # Skip the i-th word
            remaining = words[:i] + words[i+1:]
            
            # If we don't have enough words left
            if len(remaining) < k:
                result.append(0)
                continue
            
            # Use a trie-like approach to find longest common prefix
            # For each possible prefix length, count how many strings share that prefix
            max_prefix_len = 0
            
            # Group strings by their prefixes of different lengths
            for prefix_len in range(1, max(len(w) for w in remaining) + 1):
                prefix_count = defaultdict(int)
                
                for word in remaining:
                    if len(word) >= prefix_len:
                        prefix = word[:prefix_len]
                        prefix_count[prefix] += 1
                
                # Check if any prefix occurs at least k times
                found = False
                for count in prefix_count.values():
                    if count >= k:
                        max_prefix_len = prefix_len
                        found = True
                        break
                
                # If no prefix of this length occurs k times, no need to check longer prefixes
                if not found:
                    break
            
            result.append(max_prefix_len)
        
        return result