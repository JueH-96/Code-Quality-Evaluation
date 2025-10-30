import math
from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        
        # Compute Z-array
        def compute_Z(arr):
            arr_len = len(arr)
            Z = [0] * arr_len
            l, r = 0, 0
            for i in range(1, arr_len):
                if i > r:
                    length = 0
                    while i + length < arr_len and arr[i + length] == arr[length]:
                        length += 1
                    Z[i] = length
                    if length > 0:
                        l = i
                        r = i + length - 1
                else:
                    k = i - l
                    if Z[k] < r - i + 1:
                        Z[i] = Z[k]
                    else:
                        length = r - i + 1
                        p = r + 1
                        while p < arr_len and arr[p] == arr[p - i]:
                            length += 1
                            p += 1
                        Z[i] = length
                        l = i
                        r = p - 1
            return Z
        
        Z_array = compute_Z(nums)
        
        # Compute prefix hash for subarray equality checks
        BASE = 31
        MOD = 1000000007
        pow_base = [1] * (n + 1)
        for i in range(1, n + 1):
            pow_base[i] = (pow_base[i - 1] * BASE) % MOD
        prefix_hash = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_hash[i] = (prefix_hash[i - 1] * BASE + nums[i - 1]) % MOD
        
        def get_hash(L, R):
            val1 = prefix_hash[R + 1]
            prod = (prefix_hash[L] * pow_base[R - L + 1] % MOD)
            hash_val = (val1 - prod + MOD) % MOD
            return hash_val
        
        # Iterate over all possible A and B
        ans = 0
        for A in range(1, n - 1):  # A from 1 to n-2 inclusive
            len_nums1 = A
            for B in range(A, n - 1):  # B from A to n-2 inclusive
                len2 = B - A + 1
                len_nums3 = n - B - 1
                # Condition A: nums1 is prefix of nums2
                cond_A = (Z_array[A] >= A and len_nums1 <= len2)
                # Condition B: nums2 is prefix of nums3
                cond_B = False
                if len2 <= len_nums3:
                    hash_nums2 = get_hash(A, B)
                    end_B2 = (B + 1) + len2 - 1
                    hash_nums3_start = get_hash(B + 1, end_B2)
                    if hash_nums2 == hash_nums3_start:
                        cond_B = True
                if cond_A or cond_B:
                    ans += 1
        
        return ans