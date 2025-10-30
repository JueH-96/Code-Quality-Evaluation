class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        m = len(pattern)
        # Precompute the positions where each character in pattern can be matched in source
        # We need to find for each character in pattern, the earliest and latest positions in source
        # that can be used to match it, considering the order.
        
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
        # that do not interfere with the earliest and latest positions.
        
        # We can iterate through the target indices and check if removing them affects the earliest or latest positions.
        # If not, we can remove them.
        
        # To do this efficiently, we can precompute the positions that are critical for the pattern.
        # A position is critical if it is the earliest or latest position for some character in the pattern.
        
        critical = set()
        for i in range(m):
            critical.add(earliest[i])
            critical.add(latest[i])
        
        # Now, we can iterate through the target indices and count how many are not in the critical set.
        count = 0
        for idx in targetIndices:
            if idx not in critical:
                count += 1
        return count