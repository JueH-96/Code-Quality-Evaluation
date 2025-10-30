class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        def can_form_subsequence(s: str, p: str) -> bool:
            it = iter(s)
            return all(char in it for char in p)
        
        # Binary search to find the maximum number of characters we can remove
        left, right = 0, len(targetIndices)
        while left < right:
            mid = (left + right + 1) // 2
            # Create a list of characters to consider removing
            removed_indices = set(targetIndices[:mid])
            # Build the new source string after removals
            new_source = ''.join(source[i] for i in range(len(source)) if i not in removed_indices)
            # Check if pattern is still a subsequence of the new source
            if can_form_subsequence(new_source, pattern):
                left = mid  # Try removing more characters
            else:
                right = mid - 1  # Too many removed, try less
        
        return left