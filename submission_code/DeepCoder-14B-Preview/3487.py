from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        def is_subsequence(s, t):
            it = iter(t)
            return all(c in it for c in s)
        
        current_source = [(i, c) for i, c in enumerate(source)]
        removed = set()
        count = 0
        
        # Process target indices in reverse order
        for i in sorted(targetIndices, reverse=True):
            if i in removed:
                continue
            # Find the position of i in current_source
            pos = -1
            for idx, (orig_i, c) in enumerate(current_source):
                if orig_i == i:
                    pos = idx
                    break
            if pos == -1:
                continue  # i not in current_source
            
            # Remove the element
            del current_source[pos]
            removed.add(i)
            
            # Check if pattern is a subsequence
            temp_source = ''.join([c for (orig_i, c) in current_source])
            if is_subsequence(pattern, temp_source):
                count += 1
            else:
                # Revert the removal
                current_source.insert(pos, (i, source[i]))
                removed.remove(i)
        
        return count