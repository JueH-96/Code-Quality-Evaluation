class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Calculate the number of non-infected segments between sick children
        segments = []
        if sick[0] > 0:
            segments.append(sick[0])  # Segment from 0 to sick[0]-1
        
        for i in range(1, len(sick)):
            if sick[i] - sick[i-1] > 1:
                segments.append(sick[i] - sick[i-1] - 1)  # Segment between sick[i-1]+1 and sick[i]-1
        
        if sick[-1] < n-1:
            segments.append(n - 1 - sick[-1])  # Segment from sick[-1]+1 to n-1
        
        # Function to calculate factorial modulo MOD
        def factorial(x):
            result = 1
            for i in range(2, x + 1):
                result = (result * i) % MOD
            return result
        
        # Function to calculate number of ways to infect a segment of length `length`
        def ways_to_infect_segment(length):
            if length == 0:
                return 1
            return factorial(length)
        
        # Calculate the total number of infection sequences
        result = 1
        for segment_length in segments:
            result = (result * ways_to_infect_segment(segment_length)) % MOD
        
        return result