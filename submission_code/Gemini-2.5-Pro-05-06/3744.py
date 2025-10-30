import math

class Solution:
    # Class-level variables for precomputation, shared across test cases if Solution object persists.
    _precomputed_done = False
    powers_of_4 = []
    block_prefix_sum = []

    def __init__(self):
        # Instance-level memoization table for P(N) values.
        self.memo_P = {} 
        
        # Trigger precomputation if it hasn't been done yet by any instance.
        if not Solution._precomputed_done:
            Solution._do_precompute()
            Solution._precomputed_done = True
            
    @classmethod
    def _do_precompute(cls):
        # Max cost for N up to 10^9 is 15 (e.g., cost(4^14) = 15 since 4^14 is the first number with cost 15).
        # K_N (cost(N)) can be up to 15.
        # K_N-1 can be up to 14. Arrays need to cover index 14.
        # block_prefix_sum[k] stores sum for costs 1..k. So index up to 14 for K_N-1 argument.
        # powers_of_4[i] stores 4^i. Index up to 14 for 4^(K_N-1) argument.
        # Size 16 (indices 0..15) is safe and covers these needs.
        MAX_K_IDX = 15 
        
        cls.powers_of_4 = [0] * (MAX_K_IDX + 1) # Size 16, indices 0..15
        cls.block_prefix_sum = [0] * (MAX_K_IDX + 1) # Size 16, indices 0..15

        cls.powers_of_4[0] = 1 # 4^0
        for i in range(1, MAX_K_IDX + 1):
            cls.powers_of_4[i] = cls.powers_of_4[i-1] * 4

        # cls.block_prefix_sum[0] is already 0 (sum of costs for k_cost from 1 up to 0 is 0)
        # k_cost is the actual cost value (1, 2, ...)
        for k_cost_val in range(1, MAX_K_IDX + 1):
            # Number of elements x with cost(x) = k_cost_val in a full block [4^(k_cost_val-1), 4^k_cost_val - 1]
            # This count is 4^k_cost_val - 4^(k_cost_val-1) = 3 * 4^(k_cost_val-1)
            # powers_of_4[k_cost_val-1] stores 4^(k_cost_val-1)
            num_elements_in_block = 3 * cls.powers_of_4[k_cost_val-1]
            
            sum_for_this_block_cost_value = k_cost_val * num_elements_in_block
            cls.block_prefix_sum[k_cost_val] = cls.block_prefix_sum[k_cost_val-1] + sum_for_this_block_cost_value
        
    def _get_cost_of_n(self, n: int) -> int:
        if n == 0:
            return 0
        
        # cost(n) = k means n needs k divisions by 4 to become 0.
        # This is equivalent to 1 + cost(floor(n/4)).
        _n_copy = n
        k = 0
        while _n_copy > 0:
            _n_copy //= 4 # Integer division, floor(a/b)
            k += 1
        return k

    def _P_sum_cost_up_to_N(self, N: int) -> int:
        if N == 0:
            return 0
        
        if N in self.memo_P: # Use instance memoization table
            return self.memo_P[N]

        K_N = self._get_cost_of_n(N) # This is cost(N)

        # Sum of costs for elements in full blocks before N's block.
        # These are elements with costs 1, 2, ..., K_N-1.
        # This sum is block_prefix_sum[K_N-1].
        # (block_prefix_sum[k] stores sum of costs for full blocks with cost 1 up to k)
        sum_val = Solution.block_prefix_sum[K_N-1]
        
        # Add costs for elements in N's current block.
        # These elements range from 4^(K_N-1) (inclusive) up to N (inclusive). All have cost K_N.
        # The count of such elements is N - 4^(K_N-1) + 1.
        # 4^(K_N-1) is Solution.powers_of_4[K_N-1].
        first_num_in_block = Solution.powers_of_4[K_N-1]
        count_in_partial_block = N - first_num_in_block + 1
        sum_val += K_N * count_in_partial_block
        
        self.memo_P[N] = sum_val # Store in instance memoization table
        return sum_val

    def minOperations(self, queries: list[list[int]]) -> int:
        # __init__ handles one-time precomputation logic via class variables.
        # self.memo_P is initialized in __init__ and persists for this Solution object instance,
        # allowing reuse of P(N) calculations across multiple queries, even from different
        # calls to minOperations if the LeetCode test runner reuses the Solution object.
        
        grand_total_ops = 0
        for l_val, r_val in queries:
            sum_costs_for_query = self._P_sum_cost_up_to_N(r_val) - self._P_sum_cost_up_to_N(l_val - 1)
            
            # ops_for_query = ceil(sum_costs_for_query / 2)
            # Using integer arithmetic: (X + Y - 1) // Y for ceil(X/Y)
            # For Y=2, this is (sum_costs_for_query + 1) // 2
            ops_for_query = (sum_costs_for_query + 1) // 2
            
            grand_total_ops += ops_for_query
            
        return grand_total_ops