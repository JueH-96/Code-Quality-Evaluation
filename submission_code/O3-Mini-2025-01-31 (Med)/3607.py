from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # Precompute smallest prime factor (spf) for numbers up to the maximum value.
        max_val = max(nums)
        N = max_val + 1
        spf = list(range(N))
        for i in range(2, int(N ** 0.5) + 1):
            if spf[i] == i:  # i is prime
                for j in range(i * i, N, i):
                    if spf[j] == j:
                        spf[j] = i
        
        # Cache for each number's "transformation chain".
        # For a number x, its chain is defined as follows:
        # chain[0] = x, and if x is composite then operation returns x // spf[x],
        # so chain[1] = x // spf[x]. We continue until we reach a prime.
        chain_cache = {}
        def get_chain(x: int) -> List[int]:
            if x in chain_cache:
                return chain_cache[x]
            chain = [x]
            cur = x
            # For composite numbers spf[cur] < cur; for primes, spf[cur] == cur.
            while spf[cur] != cur:
                nxt = cur // spf[cur]
                chain.append(nxt)
                cur = nxt
            chain_cache[x] = chain
            return chain
        
        # We need to choose an operation count for each element so that the final 
        # array becomes non-decreasing. Notice:
        # - Each operation reduces the number (since x // spf[x] < x when x is composite).
        # - If x is prime, operating on it divides by 1 and it doesn't change; so we avoid extra ops.
        #
        # Thus for each number, we have a discrete set of possible values:
        #   chain = [x, f(x), f(f(x)), ...] until it becomes prime.
        # We want to choose, for every index i, a chain state chain[i][k_i] 
        # such that when scanning from left-to-right:
        #   chain[0][k_0] <= chain[1][k_1] <= ... <= chain[n-1][k_{n-1]]
        # and the total operations, sum(k_i), is minimized.
        #
        # A greedy way to achieve this is to process the array from rightmost to leftmost.
        # For the rightmost element, we choose 0 operations (keeping it largest), and then
        # for each preceding element, we choose the option in its chain with the fewest operations 
        # (and thus highest possible value) that is still <= the value chosen for its right neighbor.
        
        total_ops = 0
        n = len(nums)
        # For the rightmost element, best to use 0 operations.
        chosen_chain = get_chain(nums[-1])
        curr_allowed = chosen_chain[0]  # this is the rightmost chosen number
        
        # Process the rest of the array from right to left.
        for i in range(n - 2, -1, -1):
            chain = get_chain(nums[i])
            # Find the minimal k for which chain[k] <= curr_allowed.
            found = False
            for k, val in enumerate(chain):
                if val <= curr_allowed:
                    total_ops += k
                    # To maximize flexibility for earlier (left) elements, 
                    # use the largest possible final value (i.e. minimal op count).
                    curr_allowed = val
                    found = True
                    break
            if not found:
                return -1
        return total_ops

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.minOperations([25, 7]))  # Expected output: 1
    # Example 2:
    print(sol.minOperations([7, 7, 6]))  # Expected output: -1
    # Example 3:
    print(sol.minOperations([1, 1, 1, 1]))  # Expected output: 0