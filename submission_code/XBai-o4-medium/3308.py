from collections import defaultdict
class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        occurrences = defaultdict(list)
        for idx, c in enumerate(s):
            occurrences[c].append(idx)
        
        if not occurrences:
            return ""
        
        max_freq = max(len(v) for v in occurrences.values())
        
        result_indices = []
        for c in occurrences:
            if len(occurrences[c]) == max_freq:
                result_indices.append(occurrences[c][-1])
        
        result_indices.sort()
        return ''.join(s[i] for i in result_indices)