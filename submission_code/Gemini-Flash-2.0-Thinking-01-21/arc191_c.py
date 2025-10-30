import sys

def solve():
    # Read the input integer N for the current test case.
    n = int(sys.stdin.readline())

    # We need to find a pair of positive integers (A, M) such that
    # the smallest positive integer n for which A^n - 1 is a multiple of M is equal to the input N.
    # This condition is equivalent to saying that the order of A modulo M is N, i.e., ord_M(A) = N.

    # We propose the pair (A, M) = (N + 1, N^2).
    # Let's verify if this pair satisfies the conditions.

    # Condition 1: A and M are positive integers.
    # Given 1 <= N <= 10^9.
    # A = N + 1. Since N >= 1, A >= 1 + 1 = 2. A is a positive integer.
    # M = N^2. Since N >= 1, M >= 1^2 = 1. M is a positive integer.
    # This condition is satisfied.

    # Condition 2: A and M are between 1 and 10^18, inclusive.
    # Given 1 <= N <= 10^9.
    # A = N + 1. 1 + 1 <= N + 1 <= 10^9 + 1.
    # 2 <= A <= 10^9 + 1. Since 10^9 + 1 is much less than 10^18, A is within [1, 10^18].
    # M = N^2. 1^2 <= N^2 <= (10^9)^2.
    # 1 <= M <= 10^18. M is within [1, 10^18].
    # This condition is satisfied. Python's built-in integers handle numbers up to 10^18.

    # Condition 3: The smallest positive integer n such that A^n - 1 is a multiple of M is N.
    # This means ord_M(A) = N. We need to show ord_{N^2}(N + 1) = N.
    # We need the smallest positive integer n such that (N + 1)^n \equiv 1 \pmod{N^2}.

    # Using the binomial expansion, for any positive integer n:
    # (N + 1)^n = \binom{n}{0}N^0 + \binom{n}{1}N^1 + \binom{n}{2}N^2 + \dots + \binom{n}{n}N^n
    # (N + 1)^n = 1 + nN + \frac{n(n-1)}{2}N^2 + \dots + N^n

    # Consider this equation modulo N^2. All terms with $N^k$ where k >= 2 are congruent to 0 mod N^2.
    # So, (N + 1)^n \equiv 1 + nN \pmod{N^2}.

    # We are looking for the smallest positive integer n such that (N + 1)^n \equiv 1 \pmod{N^2}.
    # Using the congruence derived above, this is equivalent to finding the smallest positive integer n such that:
    # 1 + nN \equiv 1 \pmod{N^2}
    # Subtracting 1 from both sides:
    # nN \equiv 0 \pmod{N^2}

    # The congruence nN \equiv 0 \pmod{N^2} means that nN is a multiple of N^2.
    # So, nN = k \cdot N^2 for some integer k.
    # Since N is a positive integer (N >= 1), we can divide the equation by N:
    # n = k \cdot N

    # We are looking for the smallest positive integer n that satisfies n = k \cdot N for some integer k.
    # The positive values of n that are multiples of N are N, 2N, 3N, ...
    # The smallest positive integer in this set is N itself.
    # Therefore, the smallest positive integer n such that (N + 1)^n \equiv 1 \pmod{N^2} is exactly N.
    # This shows that ord_{N^2}(N + 1) = N.

    # The pair (A, M) = (N + 1, N^2) satisfies all the required conditions.

    # Calculate A and M
    a = n + 1
    m = n * n

    # Print the result for the current test case
    print(f"{a} {m}")

# Read the number of test cases
t = int(sys.stdin.readline())

# Process each test case
for _ in range(t):
    solve()