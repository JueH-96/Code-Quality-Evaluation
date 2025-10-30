class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 7:
            return 0
        
        max_val = 1000
        # Precompute prefix_counts where prefix_counts[i][v] represents the count of value v in nums[0..i-1]
        prefix_counts = [[0] * (max_val + 1) for _ in range(n + 2)]
        for i in range(n):
            current = nums[i]
            for v in range(1, max_val + 1):
                prefix_counts[i+1][v] = prefix_counts[i][v]
            if current <= max_val:
                prefix_counts[i+1][current] += 1
        
        # Precompute freq where freq[i][v] represents the count of value v in nums[i..n-1]
        freq = [[0] * (max_val + 1) for _ in range(n + 2)]
        for i in range(n-1, -1, -1):
            current = nums[i]
            for v in range(1, max_val + 1):
                freq[i][v] = freq[i+1][v]
            if current <= max_val:
                freq[i][current] += 1
        
        total = 0
        for q in range(n):
            if q < 2:
                continue  # need p <= q-2 >= 0 implies q >= 2
            for r in range(q + 2, n):
                s_start = r + 2
                if s_start >= n:
                    continue
                q_val = nums[q]
                r_val = nums[r]
                current_total = 0
                # Iterate through all possible p_val
                for p_val in range(1, max_val + 1):
                    count_p = prefix_counts[q-1][p_val]
                    if count_p == 0:
                        continue
                    product = p_val * r_val
                    if product % q_val != 0:
                        continue
                    k = product // q_val
                    if k < 1 or k > max_val:
                        continue
                    count_s = freq[r+2][k]
                    current_total += count_p * count_s
                total += current_total
        return total