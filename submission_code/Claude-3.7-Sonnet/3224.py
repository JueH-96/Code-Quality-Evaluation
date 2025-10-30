class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Precompute factorials for use in multinomial coefficient
        fact = [1]
        for i in range(1, n+1):
            fact.append((fact[-1] * i) % MOD)
        
        # Identify segments of consecutive non-infected children
        sick_set = set(sick)
        segments = []
        start = -1
        for i in range(n):
            if i not in sick_set:
                if start == -1:
                    start = i
            else:
                if start != -1:
                    segments.append((start, i-1))
                    start = -1
        if start != -1:
            segments.append((start, n-1))
        
        # Calculate valid sequences for each segment and their lengths
        segment_sequences = []
        segment_lengths = []
        for start, end in segments:
            length = end - start + 1
            segment_lengths.append(length)
            left_infected = start > 0 and start-1 in sick_set
            right_infected = end < n-1 and end+1 in sick_set
            
            if left_infected and right_infected:
                # Segment bounded by infected children on both sides
                # We have 2^(length-1) ways to infect this segment
                segment_sequences.append(pow(2, length-1, MOD))
            else:
                # Segment bounded by infected child on only one side
                # Only 1 way to infect this segment (from the infected side inward)
                segment_sequences.append(1)
        
        # Calculate the total number of valid infection sequences
        total_sequences = 1
        for seq in segment_sequences:
            total_sequences = (total_sequences * seq) % MOD
        
        if len(segment_lengths) > 1:
            # Calculate ways to interleave the segments using multinomial coefficient
            total_length = sum(segment_lengths)
            
            # Calculate multinomial coefficient using modular arithmetic
            interleaving = fact[total_length]
            for length in segment_lengths:
                # Use Fermat's Little Theorem for modular inverse
                interleaving = (interleaving * pow(fact[length], MOD-2, MOD)) % MOD
            
            total_sequences = (total_sequences * interleaving) % MOD
        
        return total_sequences