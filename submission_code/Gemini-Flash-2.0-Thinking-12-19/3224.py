class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        extended_sick = [-1] + sick + [n]
        gap_lengths = []
        for i in range(len(extended_sick) - 1):
            length = extended_sick[i+1] - extended_sick[i] - 1
            if length > 0:
                gap_lengths.append(length)
        
        num_gaps = len(gap_lengths)
        if num_gaps == 0:
            return 1 # Should not happen based on problem description, but handle for completeness
        elif num_gaps == 1:
            return gap_lengths[0] + 1
        else:
            return sum(gap_lengths)