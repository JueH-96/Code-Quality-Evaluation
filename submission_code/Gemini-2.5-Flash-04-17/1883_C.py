import sys

def solve():
    n, k = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))

    if k == 4:
        # k = 2^2. We need the total power of 2 in the product to be at least 2.
        # This is sum(nu_2(a_i)) >= 2.
        
        # Option 1: Make one a_i have nu_2 >= 2 (i.e., a multiple of 4).
        # Minimum operations for a number x to become a multiple of 4 is (4 - x % 4) % 4.
        # We find the minimum such cost over all numbers.
        m1 = 4 # Initialize with a value >= max possible ops needed for one number to be div by 4 (max is 3 for 1, 5, 9...)
        for x in a:
            rem = x % 4
            ops = (4 - rem) % 4
            m1 = min(m1, ops)

        # Option 2: Make two a_i have nu_2 >= 1 (i.e., two even numbers).
        # The total nu_2 from two even numbers is at least 1 + 1 = 2.
        # Cost to make one number even is (2 - x % 2) % 2 (which is 1 if x is odd, 0 if x is even).
        # We need the minimum total cost to make two numbers even.
        # If there are initially >= 2 even numbers, cost is 0.
        # If there is initially 1 even number, we need one more, pick an odd number (cost 1). Total cost 1.
        # If there are initially 0 even numbers, we need two more, pick two odd numbers (cost 1+1=2). Total cost 2.
        # This minimum cost is max(0, 2 - count_even_numbers).
        even_count = 0
        for x in a:
            if x % 2 == 0:
                even_count += 1
        m2 = max(0, 2 - even_count)

        # The minimum operations for k=4 is the minimum of the costs from these two options.
        print(min(m1, m2))
    else: # k is a prime number: 2, 3, or 5. k = p^1.
          # We need the total power of p in the product to be at least 1.
          # This is sum(nu_p(a_i)) >= 1.
          # This is equivalent to needing at least one number in the array to be divisible by p.
          # The minimum operations is the minimum cost to make one a_i divisible by p.
          # Minimum operations for a number x to become a multiple of k is (k - x % k) % k.
        min_ops = k # Initialize with a value >= max possible ops needed (max k-1)
        for x in a:
            rem = x % k
            ops = (k - rem) % k
            min_ops = min(min_ops, ops)
        print(min_ops)

t = int(sys.stdin.readline())
for _ in range(t):
    solve()