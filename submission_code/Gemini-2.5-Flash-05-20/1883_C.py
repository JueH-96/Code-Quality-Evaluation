import sys

def solve():
    n, k = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))

    # Initialize ans with a sufficiently large value
    ans = float('inf')

    if k == 4:
        # For k=4, we need 2 factors of 2 in the product.
        # This can be achieved by:
        # 1. Making one number divisible by 4 (0 mod 4). This contributes two factors of 2.
        # 2. Making two different numbers each contribute one factor of 2 (i.e., becoming 2 mod 4).

        # Candidate 1: Minimum operations to make one number divisible by 4
        min_ops_to_make_one_div_4 = float('inf')
        for x in a:
            # (4 - x % 4) % 4 gives the ops needed to make x a multiple of 4
            # e.g., if x % 4 == 1, ops = 3 (1->4)
            # if x % 4 == 2, ops = 2 (2->4)
            # if x % 4 == 3, ops = 1 (3->4)
            # if x % 4 == 0, ops = 0
            min_ops_to_make_one_div_4 = min(min_ops_to_make_one_div_4, (4 - x % 4) % 4)
        ans = min(ans, min_ops_to_make_one_div_4)

        # Candidate 2: Minimum operations to make two numbers become 2 mod 4
        ops_to_make_2_mod_4_costs = []
        for x in a:
            # (2 - x % 4 + 4) % 4 gives the ops needed to make x congruent to 2 mod 4
            # e.g., if x % 4 == 1, ops = 1 (1->2)
            # if x % 4 == 2, ops = 0 (2->2)
            # if x % 4 == 3, ops = 3 (3->6)
            # if x % 4 == 0, ops = 2 (0->2)
            ops_to_make_2_mod_4_costs.append((2 - x % 4 + 4) % 4)
        
        # Sort the costs to find the two smallest
        ops_to_make_2_mod_4_costs.sort()

        # If there are at least two numbers, we can combine their factors of 2
        if len(ops_to_make_2_mod_4_costs) >= 2:
            ans = min(ans, ops_to_make_2_mod_4_costs[0] + ops_to_make_2_mod_4_costs[1])
    else: # k = 2, 3, or 5 (prime numbers)
        # For prime k, the product is divisible by k if at least one number in the array is divisible by k.
        # We find the minimum operations to make any single number divisible by k.
        for x in a:
            # (k - x % k) % k gives the ops needed to make x a multiple of k
            ans = min(ans, (k - x % k) % k)
    
    sys.stdout.write(str(ans) + "
")

# Read the number of test cases
num_test_cases = int(sys.stdin.readline())

# Process each test case
for _ in range(num_test_cases):
    solve()