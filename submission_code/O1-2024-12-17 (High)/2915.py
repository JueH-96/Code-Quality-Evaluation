from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # We will use a prefix-sum + frequency map approach:
        # Let p[i] = (number of elements nums[0..i] whose value % modulo == k) mod modulo.
        #
        # A subarray nums[l..r] is interesting if:
        #   (p[r] - p[l-1]) mod modulo == k
        # This rearranges to:
        #   p[l-1] == (p[r] - k) mod modulo.
        #
        # We'll keep a running prefix sum p and a dictionary freq
        # that maps each prefix sum value to how many times it appears.
        #
        # Steps:
        # 1) Initialize p = 0, ans = 0, freq = {0: 1}, because p[-1] = 0 by convention.
        # 2) Iterate through nums. For each num:
        #    - If num % modulo == k, increment p by 1 (then take p mod modulo).
        #    - Compute target = (p - k) mod modulo.
        #    - Add freq.get(target, 0) to ans (count how many prefix sums match our needed target).
        #    - Increment freq[p] by 1.
        # 3) Return ans at the end.

        freq = {0: 1}
        p = 0
        ans = 0
        
        for num in nums:
            if num % modulo == k:
                p = (p + 1) % modulo
            target = (p - k) % modulo
            ans += freq.get(target, 0)
            freq[p] = freq.get(p, 0) + 1
        
        return ans