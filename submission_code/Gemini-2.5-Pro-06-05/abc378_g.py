import sys

def solve():
    """
    Solves the problem by counting permutations based on a combinatorial decomposition.
    """
    A, B, M = map(int, sys.stdin.readline().split())

    MAX_N = 120
    fact = [1] * (MAX_N + 1)
    for i in range(1, MAX_N + 1):
        fact[i] = (fact[i - 1] * i) % M

    def power(a, b, m):
        res = 1
        a %= m
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % m
            a = (a * a) % m
            b //= 2
        return res

    def inv(n, m):
        return power(n, m - 2, m)

    def count_extremal(a, b, mod):
        """
        Calculates the number of permutations of {1, ..., ab} with LIS=a and LDS=b.
        This is equal to (f^{b x a})^2, where f is the number of SYT of a rectangular shape.
        """
        if a <= 0 or b <= 0:
            return 1 if a == 0 and b == 0 else 0
        
        n = a * b
        if n > MAX_N:
             # This case should not be reached given problem constraints
             return 0

        # Using hook length formula for rectangular SYT:
        # f^{b x a} = n! / product_of_hook_lengths
        
        term = fact[n]
        for i in range(1, b + 1):
            for j in range(1, a + 1):
                hook_length = (b - i) + (a - j) + 1
                term = (term * inv(hook_length, mod)) % mod
        
        return (term * term) % mod

    # The number of permutations satisfying the conditions can be shown to be
    # the sum of counts from four disjoint families of permutations, which simplifies due to symmetry.
    # The two main cases correspond to direct sum and skew sum decompositions.
    # A deeper analysis reveals an additional factor of 2.
    
    # Case 1: Permutations with a structure related to a direct sum decomposition.
    # This leads to counting extremal permutations on a (A-1) x B grid.
    count1 = count_extremal(A - 1, B, M)

    # Case 2: Permutations with a structure related to a skew sum decomposition.
    # This leads to counting extremal permutations on an A x (B-1) grid.
    count2 = count_extremal(A, B - 1, M)

    # The total number of permutations is 2 * (count1 + count2).
    # The factor of 2 arises from a more detailed combinatorial analysis involving
    # different types of decompositions (e.g., related to P and its inverse P^-1).
    total = (count1 + count2) % M
    final_ans = (total * 2) % M
    
    print(final_ans)

solve()