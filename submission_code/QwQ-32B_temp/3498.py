class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pairs = []
        for i in range(n // 2):
            a = nums[i]
            b = nums[n - 1 - i]
            pairs.append((a, b))
        P = len(pairs)
        
        max_k = k
        count_d = [0] * (max_k + 1)
        for a, b in pairs:
            d = abs(a - b)
            if d <= max_k:
                count_d[d] += 1
        
        # Initialize delta array for count_1
        delta = [0] * (max_k + 2)  # indexes 0 to max_k +1
        
        for a, b in pairs:
            d = abs(a - b)
            R_a = max(k - a, a)
            R_b = max(k - b, b)
            R = max(R_a, R_b)
            
            # Add the range [0, R]
            delta[0] += 1
            if R + 1 <= max_k + 1:
                delta[R + 1] -= 1
            
            # Subtract 1 at X = d if d is within [0, R]
            if d <= R:
                delta[d] -= 1
                delta[d + 1] += 1
        
        # Compute count_1 array using prefix sum
        count_1 = [0] * (max_k + 1)
        current = 0
        for x in range(max_k + 1):
            current += delta[x]
            count_1[x] = current
        
        # Find the maximum value of (2 * count_d[x] + count_1[x])
        max_val = 0
        for x in range(max_k + 1):
            current_val = 2 * count_d[x] + count_1[x]
            if current_val > max_val:
                max_val = current_val
        
        min_changes = 2 * P - max_val
        return min_changes