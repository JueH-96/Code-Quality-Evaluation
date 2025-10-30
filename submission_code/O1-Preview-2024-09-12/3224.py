class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        sick_set = set(sick)
        # Positions of uninfected children
        uninfected = [i for i in range(n) if i not in sick_set]
        
        # If no uninfected children, return 1
        if not uninfected:
            return 1
        
        # Find uninfected segments
        segments = []
        start = None
        for idx in uninfected:
            if start is None:
                start = idx
                prev = idx
            elif idx == prev + 1:
                prev = idx
            else:
                segments.append((start, prev))
                start = idx
                prev = idx
        if start is not None:
            segments.append((start, prev))
        
        result = 1
        for seg_start, seg_end in segments:
            L = seg_end - seg_start + 1  # Length of segment
            # Check connections at ends
            left_connected = (seg_start - 1 in sick_set)
            right_connected = (seg_end + 1 in sick_set)
            if left_connected and right_connected:
                # Segment connected at both ends
                # Approximate number of sequences as 2^(L-1)
                sequences = pow(2, L - 1, MOD)
            else:
                # Segment connected at one end or disconnected
                # Build dependency graph and count valid permutations
                # For paths connected at one end, number of sequences is 1
                sequences = 1
            result = (result * sequences) % MOD
        return result