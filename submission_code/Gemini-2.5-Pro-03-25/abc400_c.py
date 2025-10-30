import math
import sys

# Define main logic within a function for clarity
def calculate_good_integers_count(N):
    """
    Calculates the number of good integers X such that 1 <= X <= N.
    A positive integer X is called a good integer if there exists a pair of 
    positive integers (a, b) such that X = 2^a * b^2.
    
    Analysis:
    Let X be a good integer. Its prime factorization must be of the form 
    X = 2^e2 * p1^ep1 * ... * pk^epk, where pi are odd primes.
    The condition X = 2^a * b^2 implies that:
    1. The exponent e2 of prime 2 must be at least 1 (since a >= 1).
    2. The exponents ep_i of all odd primes pi must be even (from the b^2 factor).

    We can characterize good integers based on the parity of the exponent e2:
    Case 1: e2 is odd. Let e2 = 2k + 1 for some k >= 0.
        X = 2^(2k+1) * (product of odd primes raised to even powers).
        X = 2 * 2^(2k) * (product of odd primes raised to even powers)
        Let s^2 = 2^(2k) * (product of odd primes raised to even powers). 
        Then s = 2^k * (product of odd primes raised to half their even powers).
        s is a positive integer.
        So X = 2 * s^2 for some positive integer s >= 1.
    
    Case 2: e2 is even. Let e2 = 2k for some k >= 1 (since e2 >= 1).
        X = 2^(2k) * (product of odd primes raised to even powers).
        Let t^2 = X.
        Then t = 2^k * (product of odd primes raised to half their even powers).
        Since k >= 1, t must be an even positive integer.
        So X = t^2 for some positive even integer t >= 2.

    Therefore, a positive integer X is good if and only if it belongs to the set 
    S1 = {2 * s^2 | s >= 1} or the set S2 = {t^2 | t >= 1, t is even}.

    We want to count the number of good integers X such that 1 <= X <= N.
    This is |(S1 U S2) intersect [1, N]|.
    By Principle of Inclusion-Exclusion: |S1 intersect [1,N]| + |S2 intersect [1,N]| - |S1 intersect S2 intersect [1,N]|.

    Count for S1: We need 2 * s^2 <= N. This means s^2 <= N/2.
    Since s >= 1, we need 1 <= s <= sqrt(N/2).
    The number of such integers s is floor(sqrt(N/2)). Let this be C1.

    Count for S2: We need t^2 <= N, where t is positive and even.
    This means t <= sqrt(N). We need to count even integers t in [1, floor(sqrt(N))].
    The positive even integers are 2, 4, 6, ...
    Let sqrtN_floor = floor(sqrt(N)). We need to count even t such that 1 <= t <= sqrtN_floor.
    The number of such t is floor(sqrtN_floor / 2). Let this be C2.

    Intersection S1 intersect S2: An integer X is in the intersection if X = 2*s^2 and X = t^2 for some s >= 1 and even t >= 2.
    This implies 2*s^2 = t^2. Let t = 2k for some k >= 1.
    Then 2*s^2 = (2k)^2 = 4k^2.
    Dividing by 2, we get s^2 = 2k^2.
    This equation implies that s^2 is even, so s must be even. Let s = 2m for some m >= 1.
    Then (2m)^2 = 2k^2 => 4m^2 = 2k^2 => 2m^2 = k^2.
    This is the same form as the original equation. This leads to an infinite descent argument, implying the only integer solution is s=0, k=0.
    Since we require s >= 1 and k >= 1, there are no positive integer solutions.
    Thus, S1 intersect S2 is empty.

    The total count is C1 + C2.
    C1 = floor(sqrt(N/2))
    C2 = floor(floor(sqrt(N)) / 2)
    
    Implementation details:
    Use integer division `//` for floor division.
    Use `math.isqrt` for integer square root, which computes floor(sqrt(x)).
    """
    
    # Calculate C1 = floor(sqrt(N/2))
    # Integer division N // 2 computes floor(N/2)
    N_half = N // 2
    # math.isqrt computes floor(sqrt(x)) for non-negative integer x.
    # Since N >= 1, N_half >= 0, so isqrt is well-defined.
    C1 = math.isqrt(N_half) 

    # Calculate C2 = floor(floor(sqrt(N)) / 2)
    # First find floor(sqrt(N))
    # N >= 1 is guaranteed, so isqrt is well-defined.
    sqrtN = math.isqrt(N)
    # Then integer division by 2 computes floor(sqrtN / 2)
    C2 = sqrtN // 2

    # The total count is C1 + C2 because the sets S1 and S2 are disjoint.
    return C1 + C2

# Read N from standard input
# Use sys.stdin.readline for potentially faster input reading compared to input()
N = int(sys.stdin.readline())

# Calculate the result using the defined function
result = calculate_good_integers_count(N)

# Print the result to standard output
print(result)