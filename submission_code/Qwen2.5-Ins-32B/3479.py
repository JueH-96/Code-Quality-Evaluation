class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        def count_substrings_with_dominant_ones(s):
            n = len(s)
            total_substrings = n * (n + 1) // 2
            non_dominant_substrings = 0
            prefix_sum = [0] * (n + 1)
            
            for i in range(n):
                prefix_sum[i + 1] = prefix_sum[i] + (1 if s[i] == '1' else -1)
            
            from collections import defaultdict
            count = defaultdict(int)
            count[0] = 1
            
            for i in range(1, n + 1):
                for k in range(1, int((prefix_sum[i] + 1) ** 0.5) + 1):
                    non_dominant_substrings += count[prefix_sum[i] - k * k]
                count[prefix_sum[i]] += 1
            
            return total_substrings - non_dominant_substrings
        
        return count_substrings_with_dominant_ones(s)