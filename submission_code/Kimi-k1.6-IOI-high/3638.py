class Solution:
    def makeStringGood(self, s: str) -> int:
        freq = [0] * 26
        for c in s:
            freq[ord(c) - ord('a')] += 1
        total_length = sum(freq)
        min_cost = total_length  # Initialize with the cost of deleting all characters
        
        for k in range(total_length + 1):
            min_vals = [min(k, f) for f in freq]
            min_vals_sorted = sorted(min_vals, reverse=True)
            
            # Compute prefix sums for min_vals_sorted to quickly get sum of first m elements
            prefix_sum = [0] * (27)  # prefix_sum[0] = 0, prefix_sum[m] = sum of first m elements
            for m in range(1, 27):
                prefix_sum[m] = prefix_sum[m-1] + min_vals_sorted[m-1]
            
            # Check all possible subset sizes (m) from 0 to 26
            for m in range(27):
                km = k * m
                current_cost = max(total_length, km) - prefix_sum[m]
                if current_cost < min_cost:
                    min_cost = current_cost
        
        return min_cost