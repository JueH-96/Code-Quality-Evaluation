MOD = 10**9 + 7

# Explanation:
# We want to count the number of permutations of [0,1,...,n-1] such that for certain prefixes (given as requirements),
# the inversion count exactly equals the specified value.
#
# A standard idea when counting inversions is to build a permutation using an insertion DP.
# When inserting an element into a permutation of length i, there are (i+1) positions where the new element can be
# inserted. Inserting at position k (0-indexed) increases the inversion count by k because it “jumps over”
# k elements that become greater than the new element.
#
# Let dp[i][inv] be the number of ways to form a permutation of length i having exactly 'inv' inversions.
# The recurrence when going from i → i+1 elements is:
#    dp[i+1][inv + k] += dp[i][inv]    for k = 0 .. i
# However, we only care about inversion counts up to 400 because every required inversion count is in [0, 400].
#
# We enforce the prefix requirements as follows:
# - For each requirement [end, cnt], the requirement applies to the prefix of length (end + 1).
#   That is, when we complete building a prefix of length (end+1), we “zero-out” all dp states except
#   the one with inversion count equal to cnt.
#
# To optimize the inner loop we recognize that the update is “convolution” by a vector of ones of length (i+1).
# Using cumulative sums allows computing:
#    new_dp[new_inv] = prefix[new_inv] - prefix[new_inv - (i+1)]   (if defined)
#
# The final answer is dp[n] corresponding to the requirement on the full permutation (n elements). 
# The input guarantees that at least one requirement has end = n-1 (meaning prefix length n).
#
# Below is the complete solution.
  
class Solution:
    def numberOfPermutations(self, n: int, requirements: list[list[int]]) -> int:
        # Build a dictionary mapping from prefix length (i+1) to required inversion count.
        req_map = {}
        for end, cnt in requirements:
            req_map[end + 1] = cnt  # Requirement for prefix [0..end] (length end+1)

        maxInv = 400
        # dp[i] will hold ways for permutation of length i for inversion counts from 0 to maxInv.
        dp = [0] * (maxInv + 1)
        dp[0] = 1
        
        # Build permutations from 0 elements to n elements.
        for i in range(0, n):
            new_dp = [0] * (maxInv + 1)
            # Build cumulative sum of dp (for faster convolution update)
            prefix_sum = [0] * (maxInv + 1)
            prefix_sum[0] = dp[0]
            for j in range(1, maxInv + 1):
                prefix_sum[j] = (prefix_sum[j - 1] + dp[j]) % MOD

            # When inserting a new element, we add between 0 and i inversions.
            # new_dp[new_inv] = sum_{k=0}^{i} dp[new_inv - k] if new_inv - k >= 0.
            for new_inv in range(maxInv + 1):
                low = new_inv - i
                if low < 0:
                    low = 0
                res = prefix_sum[new_inv]
                if low - 1 >= 0:
                    res = (res - prefix_sum[low - 1]) % MOD
                new_dp[new_inv] = res
            # If there is a requirement for prefix length i+1, enforce it.
            if (i + 1) in req_map:
                req_inv = req_map[i + 1]
                temp = [0] * (maxInv + 1)
                if req_inv <= maxInv:
                    temp[req_inv] = new_dp[req_inv]
                new_dp = temp
            dp = new_dp
        
        # At length n, there must be a requirement for prefix n (since at least one requirement has end==n-1).
        if n in req_map:
            req_inv = req_map[n]
            return dp[req_inv] % MOD if req_inv <= maxInv else 0
        else:
            return sum(dp) % MOD


# The following is used for local testing or when running as a script.
if __name__ == '__main__':
    # For standalone test run using standard input:
    import sys
    data = sys.stdin.read().strip().split()
    if data:
        it = iter(data)
        n = int(next(it))
        req_list = []
        # Expecting the rest of the data as pairs (end, cnt)
        while True:
            try:
                a = next(it)
                b = next(it)
                req_list.append([int(a), int(b)])
            except StopIteration:
                break
        sol = Solution()
        ans = sol.numberOfPermutations(n, req_list)
        sys.stdout.write(str(ans))
    
    # Example test cases:
    else:
        sol = Solution()
        print(sol.numberOfPermutations(3, [[2,2],[0,0]]))      # Expected output: 2
        print(sol.numberOfPermutations(3, [[2,2],[1,1],[0,0]]))  # Expected output: 1
        print(sol.numberOfPermutations(2, [[0,0],[1,0]]))        # Expected output: 1