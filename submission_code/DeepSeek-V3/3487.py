class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        m = len(pattern)
        # Precompute the positions where each character in pattern can be matched in source
        # We need to find for each character in pattern, the earliest and latest positions in source
        # that can be used to match it.
        
        # Precompute the earliest possible positions for each character in pattern
        earliest = [0] * m
        ptr = 0
        for i in range(n):
            if ptr < m and source[i] == pattern[ptr]:
                earliest[ptr] = i
                ptr += 1
        
        # Precompute the latest possible positions for each character in pattern
        latest = [0] * m
        ptr = m - 1
        for i in range(n-1, -1, -1):
            if ptr >= 0 and source[i] == pattern[ptr]:
                latest[ptr] = i
                ptr -= 1
        
        # Now, for each target index, check if it can be removed without breaking the subsequence
        # We need to ensure that for each character in pattern, the earliest and latest positions
        # are still valid after removing the target indices.
        
        # To maximize the number of operations, we should try to remove as many target indices as possible
        # that do not affect the earliest and latest positions of the pattern characters.
        
        # We can iterate through the target indices and for each, check if it can be removed.
        # To do this, we need to ensure that the target index is not in any of the earliest or latest positions.
        
        # However, since the target indices are sorted, we can process them in order and keep track of the
        # current state of the source after each removal.
        
        # To simplify, we can precompute for each target index whether it can be removed.
        # A target index can be removed if it is not in any of the earliest or latest positions.
        
        # Create a set of the earliest and latest positions
        critical_positions = set(earliest + latest)
        
        # Now, iterate through the target indices and count how many are not in critical_positions
        count = 0
        for idx in targetIndices:
            if idx not in critical_positions:
                count += 1
        
        return count