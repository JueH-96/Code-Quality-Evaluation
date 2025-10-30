from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = n // 2
        pairs = []
        for i in range(m):
            a = nums[i]
            b = nums[n - 1 - i]
            pairs.append((a, b))
        
        max_possible = k
        count_original = [0] * (max_possible + 2)
        freq_ma = [0] * (max_possible + 2)
        freq_mb = [0] * (max_possible + 2)
        freq_mpair = [0] * (max_possible + 2)
        
        for a, b in pairs:
            d = abs(a - b)
            if d <= max_possible:
                count_original[d] += 1
            
            ma = max(a, k - a)
            mb = max(b, k - b)
            freq_ma[ma] += 1
            freq_mb[mb] += 1
            
            mpair = min(ma, mb)
            freq_mpair[mpair] += 1
        
        # Compute suffix sums for freq_mb, freq_ma, and freq_mpair
        suffix_mb = [0] * (max_possible + 2)
        suffix_mb[max_possible] = freq_mb[max_possible]
        for i in range(max_possible - 1, -1, -1):
            suffix_mb[i] = suffix_mb[i + 1] + freq_mb[i]
        
        suffix_ma = [0] * (max_possible + 2)
        suffix_ma[max_possible] = freq_ma[max_possible]
        for i in range(max_possible - 1, -1, -1):
            suffix_ma[i] = suffix_ma[i + 1] + freq_ma[i]
        
        suffix_mpair = [0] * (max_possible + 2)
        suffix_mpair[max_possible] = freq_mpair[max_possible]
        for i in range(max_possible - 1, -1, -1):
            suffix_mpair[i] = suffix_mpair[i + 1] + freq_mpair[i]
        
        min_changes = float('inf')
        for X in range(0, k + 1):
            current_original = count_original[X] if X <= max_possible else 0
            cs1 = suffix_mb[X] if X <= max_possible else 0
            cs2 = suffix_ma[X] if X <= max_possible else 0
            cs1_and_cs2 = suffix_mpair[X] if X <= max_possible else 0
            count_S1_or_S2 = cs1 + cs2 - cs1_and_cs2
            total = 2 * m - current_original - count_S1_or_S2
            if total < min_changes:
                min_changes = total
        
        return min_changes