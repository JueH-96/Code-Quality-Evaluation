from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0

        # Compute the Z array of nums.
        # Z[i] = length of the longest segment starting at i which is equal to the prefix starting at 0.
        z = [0] * n
        l, r = 0, 0
        for i in range(1, n):
            if i <= r:
                z[i] = min(r - i + 1, z[i - l])
            while i + z[i] < n and nums[z[i]] == nums[i + z[i]]:
                z[i] += 1
            if i + z[i] - 1 > r:
                l, r = i, i + z[i] - 1

        # We will use rolling hash to compare subarrays in O(1).
        mod = 10**9 + 7
        base = 131
        # H will store prefix hashes: H[i] is the hash of nums[0:i].
        H = [0] * (n + 1)
        P = [1] * (n + 1)
        for i in range(n):
            H[i+1] = (H[i] * base + (nums[i] + 1)) % mod
            P[i+1] = (P[i] * base) % mod

        # Helper inline: get the hash of subarray nums[l:r+1].
        # Rather than writing a separate function (which incurs function call overhead),
        # we will inline the computation when needed:
        #    hash = (H[r+1] - H[l] * P[r-l+1]) % mod

        # Count splits satisfying condition A:
        # (nums1 is a prefix of nums2)
        # If we split as nums1 = nums[:i], nums2 = nums[i:j] and nums3 = nums[j:],
        # condition A requires i <= (j - i) (i.e. j >= 2*i)
        # and also nums[0:i] equals nums[i:2*i] (which is exactly: z[i] >= i).
        countA = 0
        # For a given i, if the first condition holds, any j with j in [2*i, n) makes a valid split.
        # (Remember j must be at most n-1 because nums3 must be non-empty.)
        for i in range(1, n):
            if 2 * i < n and z[i] >= i:
                countA += (n - 2 * i)

        # Count splits satisfying condition B:
        # (nums2 is a prefix of nums3)
        # For split (i, j) with nums1 = nums[:i], nums2 = nums[i:j], nums3 = nums[j:],
        # condition B means that (j - i) <= (n - j)  [so there is room in nums3]
        # and the first (j - i) elements of nums3 equal nums2; that is,
        # nums[i:j] == nums[j:j+(j-i)].
        # For each fixed i (1 <= i <= n-2), let L = j - i.
        # Then j is valid if  1 <= L <= (n - i) // 2 and:
        #    nums[i:i+L] equals nums[i+L:i+2*L].
        countB = 0
        # We also want to count the number of splits that satisfy both conditions.
        countOverlap = 0  # splits that satisfy both A and B.
        
        # We'll iterate over possible starting cut i for nums2,
        # then let L = length(nums2) vary from 1 up to (n-i)//2.
        # For each L, j = i + L; condition B is true when the two segments of length L match.
        # If in addition i qualifies for condition A (that is: 2*i < n and z[i] >= i)
        # and L is at least i (which ensures j = i+L >= 2*i) then the split also satisfies condition A.
        m = mod  # local alias for mod
        for i in range(1, n - 1):
            maxL = (n - i) // 2   # maximum possible L so that i+2*L <= n
            condA = (2 * i < n and z[i] >= i)
            # Loop over candidate lengths L for nums2.
            for L in range(1, maxL + 1):
                # Compute hash for nums[i:i+L]
                h1 = (H[i + L] - H[i] * P[L]) % m
                # Compute hash for nums[i+L:i+2*L]
                h2 = (H[i + 2 * L] - H[i + L] * P[L]) % m
                if h1 == h2:
                    countB += 1
                    if condA and L >= i:
                        countOverlap += 1

        # The total number of beautiful splits is given by the union of those splits.
        # (Splits may satisfy both conditions so we subtract the overlap.)
        return countA + countB - countOverlap