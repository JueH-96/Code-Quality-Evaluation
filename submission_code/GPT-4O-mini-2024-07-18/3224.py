class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Calculate the gaps between infected children
        gaps = []
        prev = -1
        
        for position in sick:
            if prev + 1 < position:
                gaps.append(position - prev - 1)
            prev = position
        
        # Check for the gap after the last infected child
        if prev < n - 1:
            gaps.append(n - prev - 1)
        
        # Calculate the number of ways to infect children in each gap
        result = 1
        
        for gap in gaps:
            # Each gap can be filled in (gap + 1) ways
            result = result * (gap + 1) % MOD
        
        return result