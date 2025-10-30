import math

def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    if k == 2 or k == 3 or k == 5:
        # For prime k, we need one a_i to be divisible by k.
        # Min operations is min over a_i of (ops to make a_i div by k).
        min_ops_for_k = k  # Initialize with a value >= k-1
        for x_val in a:
            if x_val % k == 0:
                min_ops_for_k = 0
                break
            # Cost to make x_val divisible by k is (k - (x_val % k))
            min_ops_for_k = min(min_ops_for_k, k - (x_val % k))
        print(min_ops_for_k)
    elif k == 4:
        # ops_to_make_one_num_div_by_4: min cost to make one a_i divisible by 4
        ops_to_make_one_num_div_by_4 = 4  # Max cost for one num is 3 (e.g. 1->4)

        # S2_factors: count of factors of 2 from array elements, capped at 2.
        S2_factors = 0
        
        num_odd_numbers = 0

        for x_val in a:
            # Cost to make current x_val divisible by 4
            cost_x_div_by_4 = (4 - (x_val % 4)) % 4 
            ops_to_make_one_num_div_by_4 = min(ops_to_make_one_num_div_by_4, cost_x_div_by_4)

            if x_val % 4 == 0:
                S2_factors += 2
            elif x_val % 2 == 0: # x_val % 4 == 2
                S2_factors += 1
            else: # x_val is odd
                num_odd_numbers += 1
        
        S2_factors = min(S2_factors, 2)

        if S2_factors == 2: # Product already divisible by 4
            print(0)
        elif S2_factors == 1: # Need one more factor of 2
            # Option A: Make one element div by 4. Cost ops_to_make_one_num_div_by_4.
            # Option B: If an odd number exists, make it even. Cost 1.
            cost_odd_to_even = float('inf')
            if num_odd_numbers > 0:
                cost_odd_to_even = 1
            
            print(min(ops_to_make_one_num_div_by_4, cost_odd_to_even))
        else: # S2_factors == 0. All numbers are odd. Need two factors of 2.
            # Option A: Make one element div by 4. Cost ops_to_make_one_num_div_by_4.
            # Option B: Make two odd numbers even. Cost 2 (if N >= 2).
            cost_two_odds_to_evens = float('inf')
            if n >= 2: # Since S2_factors is 0, all n numbers are odd.
                       # If n >= 2, we can pick two to make even.
                cost_two_odds_to_evens = 2
            
            print(min(ops_to_make_one_num_div_by_4, cost_two_odds_to_evens))

num_test_cases = int(input())
for _ in range(num_test_cases):
    solve()