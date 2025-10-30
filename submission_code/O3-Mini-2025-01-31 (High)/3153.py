from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        # A key observation: The given operation preserves for every bit position b
        # the total number of ones present over all numbers.
        # In any change, when you replace a and b by (a AND b) and (a OR b),
        # the count of ones in each binary digit remains unchanged.
        #
        # This means that for each bit position b (with value 2^b) the total number of ones
        # is invariant. Let cnt[b] = how many numbers in the original array have bit b set.
        #
        # In our final configuration (after doing arbitrarily many operations)
        # we want to choose k numbers and maximize the sum of their squares.
        # Because the square function is convex, it is best to “concentrate”
        # the ones into a few numbers. However, note that within one number a given
        # bit can only appear once.
        #
        # Thus for each bit b we can use that 1 at most once in any one number,
        # and across the k chosen numbers we can have at most min(cnt[b], k) copies of bit b.
        # Imagine that for bit b we have “copies” (each copy contributes 2^b) and
        # you are allowed to assign these copies to different chosen numbers but at most one copy per number.
        #
        # The best way to maximize the sum of squares is to “coordinate” the assignment
        # so that the chosen numbers become nested. In other words, there exists an optimal configuration
        # where if we sort the chosen numbers in non‐increasing order and denote them as:
        #    X₁, X₂, …, Xₖ,
        # then for each bit b with value 2^b, if we let r = min(cnt[b], k), we have that
        # the top r numbers get that bit and the remaining (k - r) numbers do not.
        # That is:
        #    X₁ = Σ{ for all bits b with cnt[b] ≥ 1 } 2^b,
        #    X₂ = Σ{ for all bits b with cnt[b] ≥ 2 } 2^b,
        #    ⋯,
        #    Xₖ = Σ{ for all bits b with cnt[b] ≥ k } 2^b.
        #
        # Then the final answer equals
        #    X₁² + X₂² + … + Xₖ²   (mod MOD).
        #
        # Now, since each bit b can appear at most in min(cnt[b], k) numbers, define:
        #    r[b] = min(cnt[b], k).
        # For b from 0 to B-1 (where B = 31 is enough since 2^30 > 10^9),
        # we then have the optimal bucket values:
        #    For i = 1 to k, let
        #         Xᵢ = Σ{ over all bits b with r[b] ≥ i } 2^b.
        #
        # We simply need to compute these Xᵢ and then compute the sum of their squares modulo MOD.
        
        B = 31  # we consider bits 0..30
        
        # Count ones in each bit position
        bitCount = [0] * B
        for num in nums:
            # Check each bit from 0 to 30.
            for b in range(B):
                if num & (1 << b):
                    bitCount[b] += 1

        # For each bit b, we can use it in at most min(bitCount[b], k) numbers.
        r = [min(bitCount[b], k) for b in range(B)]
        
        # The optimal configuration (after allowed operations) is to assign for each bit b:
        # add a copy (of value 2^b) to the top r[b] numbers.
        # Thus, if we sort the chosen numbers in descending order (or think of them as positions 1..k):
        #   For i = 1 to k, define:
        #         Xᵢ = Σ{ over all bits b where r[b] ≥ i } 2^b.
        #
        # Then the answer is sum(Xᵢ²) for i = 1 to k.
        buckets = [0] * k
        for b in range(B):
            v = 1 << b
            countCopies = r[b]  # we can put this bit in the first 'countCopies' numbers
            # Assign this bit (value v) to the top countCopies buckets.
            for i in range(countCopies):
                buckets[i] += v
                
        # Compute the final answer: sum of squares of all bucket values.
        ans = 0
        for x in buckets:
            ans = (ans + (x % MOD) * (x % MOD)) % MOD
        return ans

# For local testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.maxSum([2,6,5,8], 2))  # Expected output: 261
    # Example 2:
    print(sol.maxSum([4,5,4,7], 3))  # Expected output: 90