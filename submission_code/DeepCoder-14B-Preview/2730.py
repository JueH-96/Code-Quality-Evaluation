class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Precompute the original set bits for each x
        original_bits = []
        for x in nums:
            bits = []
            for j in range(30, -1, -1):
                if (x >> j) & 1:
                    bits.append(j)
            original_bits.append(bits)
        
        current_shifts = [0] * n
        result = 0
        
        for i in range(30, -1, -1):
            min_s = float('inf')
            best_x = -1
            for idx in range(n):
                x_bits = original_bits[idx]
                current_shift = current_shifts[idx]
                
                # Calculate possible s_candidate for each j in x_bits
                for j in x_bits:
                    s_candidate = i - j - current_shift
                    if s_candidate >= 0:
                        # Check if applying s_candidate shifts is possible within the max shift limit (15)
                        if current_shift + s_candidate <= 15:
                            if s_candidate < min_s:
                                min_s = s_candidate
                                best_x = idx
            if best_x != -1 and min_s <= k:
                result |= (1 << i)
                k -= min_s
                current_shifts[best_x] += min_s
        
        return result