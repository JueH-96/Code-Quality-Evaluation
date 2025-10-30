class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pairs = n // 2
        # baseline: if we changed both numbers in every pair:
        base = 2 * pairs
        
        # We will use a difference (or prefix sum) array “change” to add
        # a saving of 1 for each candidate X if the pair can be fixed with 1 change.
        # We'll cover values X in the range [0, k]. Use an array of length k+2.
        diff = [0] * (k + 2)
        # Also count extra saving (i.e. 1 saving) if X equals the natural difference.
        freq = [0] * (k + 1)
        
        for i in range(pairs):
            a = nums[i]
            b = nums[n - i - 1]
            lo = min(a, b)
            hi = max(a, b)
            d = hi - lo
            freq[d] += 1
            # one-change saving is possible for candidate X if X <= threshold.
            # For this pair, threshold = max(hi, k - lo)
            threshold = hi if hi >= (k - lo) else (k - lo)
            # For X in [0, threshold] one change is enough.
            diff[0] += 1
            if threshold + 1 <= k:
                diff[threshold + 1] -= 1
        
        # Build prefix sum for one-change saving for each candidate X.
        best = float('inf')
        running = 0
        for X in range(0, k+1):
            running += diff[X]
            # Total cost if candidate difference is X:
            # base cost (2 per pair) minus one-change saving (if X<=threshold) 
            # minus extra saving for pairs that already have difference equal to X.
            total_change = base - (running + freq[X])
            if total_change < best:
                best = total_change
        return best