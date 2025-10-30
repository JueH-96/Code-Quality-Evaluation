from typing import List

MOD = 10**9 + 7

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        mod = MOD
        n = len(nums)
        nums.sort()
        # Our plan:
        # Let F(L) = sum_{r=0}^{min(k-1,L)} C(L, r) mod mod.
        # Then note:
        #   • The sum of the minimums over all subsequences (of size 1..k)
        #     is ∑_{i=0}^{n-1} nums[i] * F(n-1-i)  [because if nums is sorted,
        #       the number of subsequences with minimum at index i equals
        #       sum_{r=0}^{k-1} C(n-i-1, r)]
        #   • Similarly, the sum of the maximums is ∑_{j=0}^{n-1} nums[j] * F(j)
        # Thus, the answer = ∑_{i=0}^{n-1} nums[i]*(F(i) + F(n-1-i)).
        #
        # We now want to compute F(L) for L=0,...,n-1.
        # When L < k the sum is 2^L because we are summing C(L,0)+C(L,1)+...+C(L,L).
        # For L >= k, we need to sum only to r=k-1.
        # A recurrence works well here:
        #     F(L+1) = 2 * F(L) - C(L, k-1)
        # and if we let G(L) = C(L, k-1), then
        #     G(L+1) = G(L) * (L+1) / (L - k + 2)
        # with the base value G(k-1) = C(k-1, k-1) = 1.
    
        F = [0] * n
        # For L from 0 to min(n-1, k-1): we have F(L) = 2^L.
        for L in range(min(n, k)):
            F[L] = pow(2, L, mod)
            
        # For L >= k, use the recurrence.
        # Start from L = k-1 (which is computed already) with G = C(k-1, k-1) = 1.
        if n > k:
            G = 1  # G represents C(L, k-1) for current L; initially L = k-1.
            for L in range(k - 1, n - 1):  # We want to compute F[L+1] for L from k-1 to n-2.
                F[L + 1] = (2 * F[L] - G) % mod
                # Update G to be C(L+1, k-1):
                # Using the relation: C(L+1, k-1) = C(L, k-1) * (L+1)/(L - k + 2)
                num = L + 1
                den = L - k + 2  # This is always positive here.
                inv_den = pow(den, mod - 2, mod)
                G = (G * num) % mod
                G = (G * inv_den) % mod
        
        # Now sum the contributions.
        total = 0
        for i in range(n):
            # For maximums: index i contributes with count F(i)
            # For minimums: index i (when counted from left) is the minimum in subsequences
            #   if it is at position i in sorted order. The count is F(n-1-i).
            total = (total + nums[i] * (F[i] + F[n - 1 - i])) % mod
        return total

# Uncomment the following to run some tests:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.minMaxSums([1, 2, 3], 2))  # Expected output: 24
    # Example 2:
    print(sol.minMaxSums([5, 0, 6], 1))  # Expected output: 22
    # Example 3:
    print(sol.minMaxSums([1, 1, 1], 2))  # Expected output: 12