class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        def is_unique(substr, index):
            return all(substr not in s for i, s in enumerate(arr) if i != index)
        
        def find_shortest_unique(s, index):
            for length in range(1, len(s) + 1):
                candidates = []
                for i in range(len(s) - length + 1):
                    substr = s[i:i+length]
                    if is_unique(substr, index):
                        candidates.append(substr)
                if candidates:
                    return min(candidates)
            return ""
        
        return [find_shortest_unique(s, i) for i, s in enumerate(arr)]