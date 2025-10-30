class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        """
        We want to count the number of quadruples (p, q, r, s) with p < q < r < s
        and each pair separated by at least one element, i.e.:
            q - p > 1, r - q > 1, s - r > 1
        such that
            nums[p] * nums[r] == nums[q] * nums[s].
        
        Concretely, the gaps imply:
            p + 2 <= q, q + 2 <= r, r + 2 <= s.
        
        A direct O(n^4) approach is too large (n up to 1000 → 10^12 operations).
        Instead, we can do an O(n^3) solution:
        
        1) We fix r (the third position in the subsequence). r must allow s ≥ r+2
           and also p ≤ r-4 (since p+2 ≤ q ≤ r-2 is needed). Hence r goes from
           2 to n-3.
        
        2) For each r, we precompute a frequency array (freq) of nums[s] for
           s in [r+2 .. n-1]. This will let us quickly count how many valid s
           satisfy nums[q] * nums[s] = nums[p] * nums[r].
        
        3) Then, for each p in [0 .. r-4] and for each q in [p+2 .. r-2],
           let product = nums[p] * nums[r]. If product % nums[q] == 0, let needed
           = product // nums[q]. The count of such s is freq[needed].
           Accumulate this in our answer.
        
        Steps to implement:
        - Build freqForR[r], which is a list (since nums[i] ≤ 1000) tracking how
          many times each value from 1..1000 appears in nums[r+2..].
        - Then the triple loop over r, p, q to sum up valid subsequences.

        The overall complexity is O(n^3), which should be borderline but feasible
        for n=1000 in optimized Python or faster environments.
        """

        n = len(nums)
        # Max value of nums[i] is 1000, so we can store frequencies in a list of length 1001
        # freqForR[r][v] = how many times 'v' appears in nums[r+2..]
        freqForR = [[0] * 1001 for _ in range(n+1)]

        # Build frequency arrays from the right.
        # For r >= n-2, we have no s in [r+2..], so freqForR[r] stays all zeros.
        # We'll fill from r = n-3 down to 0.
        for r in range(n-3, -1, -1):
            # Copy freqForR[r+1] into freqForR[r]
            for v in range(1001):
                freqForR[r][v] = freqForR[r+1][v]
            # Now include nums[r+2] if r+2 < n
            if r+2 < n:
                freqForR[r][nums[r+2]] += 1

        ans = 0

        # Outer loop over r in [2.. n-3]
        for r in range(2, n-2):
            freq = freqForR[r]  # frequency array for s in [r+2..]
            # p can go up to r-4 at most, so p in [0.. r-4]
            # We'll just do range(0, r-3+1) and Python handles empty ranges if r-3<0.
            for p in range(0, r-3+1):
                product = nums[p] * nums[r]
                # q in [p+2.. r-2]
                for q in range(p+2, r-1):
                    if product % nums[q] == 0:
                        needed = product // nums[q]
                        if 1 <= needed <= 1000:
                            ans += freq[needed]

        return ans