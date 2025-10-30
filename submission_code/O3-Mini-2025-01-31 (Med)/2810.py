from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # Imagine that you already “own” a magic machine.
        # Initially, the chocolate of index i is of type i.
        # In one global operation (cost = x), the machine rotates all chocolate types:
        # every chocolate’s type becomes (previous type + 1) mod n.
        #
        # You want to eventually collect one chocolate of every type (0, 1, …, n-1).
        # When you "collect" a chocolate, you pay the cost assigned to that chocolate.
        #
        # Since you can perform an arbitrary number of operations (each costing x),
        # you can wait to have a chocolate “appear” at a different type.
        # Precisely, if you buy the chocolate originally at index i at a moment after
        # k operations, its type will be (i + k) mod n.
        #
        # Now observe that since you are allowed to “buy” chocolates at different times —
        # that is, at various numbers of operations performed — you can conceptually
        # decide for each chocolate type j the number of operations d (with 0 ≤ d ≤ k)
        # to “wait” so that you obtain type j as cheaply as possible.
        #
        # In particular, note that if you decide that the maximum number of operations you’ll perform is k,
        # then for each type j the cheapest way to get that type is to “look back”
        # at the available chocolates that eventually become type j.
        # A chocolate originally at index i gives type j at rotation d if (i + d) mod n = j,
        # i.e. i = (j - d) mod n. And its price is still nums[i].
        #
        # So, for a fixed overall number of operations k (thus paying k*x in operations cost),
        # for type j you can get it for
        #      cost_j(k) = min{ nums[(j-d) mod n]  for d = 0, 1, …, k }.
        #
        # And overall total cost candidate is:
        #      candidate = k*x + sum_{j=0}^{n-1} cost_j(k).
        #
        # Your answer is the minimum candidate over k = 0, 1, …, n-1.
        #
        # Notice that if k = 0, you are buying all chocolates at their initial type,
        # so total cost = sum(nums) — i.e. no operations.
        
        # Let best[j] be the best (smallest) cost to cover type j using a wait (rotation) d ≤ current k.
        best = nums[:]   # k = 0: for type j the only candidate is nums[j] (since (j-0) mod n = j)
        ans = sum(best)  # candidate when doing 0 operations
        
        # Try k = 1 to n-1 operations.
        for k in range(1, n):
            sumb = 0
            # For each type j, consider the new candidate candidate from chocolate at index (j - k) mod n.
            # best[j] stores the minimum over d = 0 .. k-1 from earlier iterations.
            # Now update best[j] = min(best[j], nums[(j-k) mod n]).
            for j in range(n):
                candidate = nums[(j - k) % n]
                if candidate < best[j]:
                    best[j] = candidate
                sumb += best[j]
            ans = min(ans, sumb + k * x)
        
        return ans


# The following is a simple driver that reads from standard input.
# It supports inputs like:
#   nums = [20,1,15], x = 5
# or space‐separated numbers.
if __name__ == '__main__':
    import sys, re
    data = sys.stdin.read().strip()
    if not data:
        # Some basic tests if no input is given.
        sol = Solution()
        print(sol.minCost([20,1,15], 5))  # Expected output: 13
        print(sol.minCost([1,2,3], 4))    # Expected output: 6
    else:
        # Try to extract the list and the value x.
        nums_match = re.search(r'\[([0-9,\s]+)\]', data)
        if not nums_match:
            raise ValueError("Could not parse nums list")
        nums_str = nums_match.group(1)
        nums = list(map(int, nums_str.split(',')))
        # Look for "x = <number>"
        x_match = re.search(r'x\s*=\s*([0-9]+)', data)
        if x_match:
            x = int(x_match.group(1))
        else:
            # If not found, assume the last token is x.
            tokens = data.split()
            x = int(tokens[-1])
        sol = Solution()
        sys.stdout.write(str(sol.minCost(nums, x)))