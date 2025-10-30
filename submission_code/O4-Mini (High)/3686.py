from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0

        # Z-algorithm: for an array arr, Z[i] = length of longest prefix of arr
        # that starts at i and matches the prefix of arr.
        def compute_z(arr: List[int]) -> List[int]:
            m = len(arr)
            z = [0] * m
            l = r = 0
            for i in range(1, m):
                if i <= r:
                    k = i - l
                    # we can copy a previously computed Z-value up to the remaining window
                    if z[k] < r - i + 1:
                        z[i] = z[k]
                    else:
                        # extend match beyond r
                        j = r + 1
                        while j < m and arr[j] == arr[j - i]:
                            j += 1
                        z[i] = j - i
                        l = i
                        r = j - 1
                else:
                    # no window, do direct comparison
                    j = 0
                    while i + j < m and arr[j] == arr[i + j]:
                        j += 1
                    z[i] = j
                    if j > 0:
                        l = i
                        r = i + j - 1
            return z

        # Precompute Z for the entire nums for condition A
        Z0 = compute_z(nums)
        result = 0

        # Condition A: nums1 is prefix of nums2
        #   i = length of nums1
        #   j >= 2*i  => 2*i <= n-1  => i <= (n-1)//2
        #   and we need Z0[i] >= i for prefix match
        #   for each such i, j can range from 2*i to n-1 inclusive => adds (n-1 - 2*i + 1) = n - 2*i splits
        max_i = (n - 1) // 2
        for i in range(1, max_i + 1):
            if Z0[i] >= i:
                result += (n - 2 * i)

        # Condition B: nums2 is prefix of nums3
        #   let i = length of nums1, k = length of nums2. Then j = i + k.
        #   we require k <= n - j = n - (i + k)  => 2*k <= n - i  => k <= (n - i)//2
        #   and the substring nums[i:i+k] must match nums[j:j+k] => lcp(i, i+k) >= k
        # We compute a Z-array for each suffix nums[i:], then Z_i[k] = lcp between
        # suffix starting at i and at i+k. We count k from 1 to (n-i)//2.
        for i in range(1, n - 1):
            max_k = (n - i) // 2
            if max_k < 1:
                continue
            suf = nums[i:]
            Zi = compute_z(suf)
            # count all k in [1..max_k] with Zi[k] >= k
            for k in range(1, max_k + 1):
                if Zi[k] >= k:
                    result += 1

        return result