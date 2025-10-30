class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        # Sort targetIndices in reverse order to try removing higher indices first
        targetIndices.sort(reverse=True)
        removed = set()
        count = 0
        
        for idx in targetIndices:
            # Tentatively remove the current index
            removed.add(idx)
            
            # Check if pattern is still a subsequence after this removal
            i = j = 0
            while i < len(source) and j < len(pattern):
                if i in removed:
                    i += 1
                    continue
                if source[i] == pattern[j]:
                    j += 1
                i += 1
            
            if j == len(pattern):
                # The pattern is still a subsequence, keep the removal
                count += 1
            else:
                # Undo the removal if it breaks the subsequence
                removed.remove(idx)
        
        return count