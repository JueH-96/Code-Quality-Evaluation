class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        from collections import defaultdict
        n = len(arr)
        mapping = defaultdict(set)  # substring -> set of indices where it occurs
        
        # Build the substring mapping
        for idx, s in enumerate(arr):
            length = len(s)
            substrings = set()
            for l in range(1, length+1):
                for i in range(length - l + 1):
                    sub = s[i:i+l]
                    mapping[sub].add(idx)
                    
        answer = [''] * n
        
        for idx, s in enumerate(arr):
            found = False
            length = len(s)
            for l in range(1, length+1):
                candidates = []
                for i in range(length - l + 1):
                    sub = s[i:i+l]
                    if mapping[sub] == {idx}:
                        candidates.append(sub)
                if candidates:
                    answer[idx] = min(candidates)
                    found = True
                    break  # Found the shortest substrings
            # If no substring found, answer[idx] remains ''
        return answer