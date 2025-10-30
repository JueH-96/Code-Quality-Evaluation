class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        ones = [i for i, num in enumerate(nums) if num == 1]
        m = len(ones)
        ones.sort()
        prefix = [0] * (m + 1)
        for i in range(m):
            prefix[i + 1] = prefix[i] + ones[i]
        res = float('inf')
        
        # Iterate over all possible s: number of existing ones used
        s_min = max(0, k - maxChanges)
        s_max = min(m, k)
        for s in range(s_min, s_max + 1):
            if s == 0:
                current = 2 * k
            else:
                mid = (s - 1) // 2
                median = ones[mid]
                # Calculate sum of distances to the median for the first s elements
                left_sum = median * (mid + 1) - (prefix[mid + 1] - prefix[0])
                right_sum = (prefix[s] - prefix[mid + 1]) - median * (s - mid - 1)
                current = left_sum + right_sum
            # Add the cost of created ones
            current += 2 * (k - s)
            if current < res:
                res = current
        
        return res if res != float('inf') else -1