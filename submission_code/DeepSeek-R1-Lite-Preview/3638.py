class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter
        freq = Counter(s)
        freq_list = [freq.get(chr(ord('a') + i), 0) for i in range(26)]
        max_freq = max(freq_list)
        
        min_operations = float('inf')
        
        # Iterate through possible target frequencies T from 0 to max_freq
        for T in range(max_freq + 1):
            shift_excess = 0
            operations = 0
            for i in range(26):
                adjusted_freq = freq_list[i] + shift_excess
                if adjusted_freq > T:
                    excess = adjusted_freq - T
                    shift_excess = excess
                    operations += excess
                elif adjusted_freq < T:
                    deficit = T - adjusted_freq
                    if shift_excess > 0:
                        fill = min(shift_excess, deficit)
                        shift_excess -= fill
                        insertions = deficit - fill
                        operations += insertions
                    else:
                        operations += deficit
                else:
                    shift_excess = 0
            # After processing all characters, delete any remaining excess
            operations += shift_excess
            min_operations = min(min_operations, operations)
        
        return min_operations