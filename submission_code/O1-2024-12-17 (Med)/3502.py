class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        # Total number of substrings
        total_sub = n * (n + 1) // 2
        
        # Count how many substrings have all character frequencies strictly less than k
        def count_substrings_with_max_freq_less_than_k(s, k):
            freq = [0] * 26
            l = 0
            max_freq = 0
            count = 0
            
            for r in range(n):
                idx_r = ord(s[r]) - ord('a')
                freq[idx_r] += 1
                # Update max frequency if needed
                if freq[idx_r] > max_freq:
                    max_freq = freq[idx_r]
                
                # While we have a character repeated k or more times, shrink from left
                while max_freq >= k and l <= r:
                    idx_l = ord(s[l]) - ord('a')
                    old_val = freq[idx_l]
                    freq[idx_l] -= 1
                    l += 1
                    # If we just removed a character that was at the max frequency,
                    # recalculate max frequency in O(26) time
                    if old_val == max_freq:
                        max_freq = max(freq)
                
                # All substrings ending at r and starting at any index in [l..r]
                # have max frequency < k
                count += (r - l + 1)
            
            return count
        
        # Number of substrings whose max frequency of a character is < k
        no_freq_at_least_k = count_substrings_with_max_freq_less_than_k(s, k)
        
        # We want the number of substrings where at least one character
        # appears k times or more
        return total_sub - no_freq_at_least_k