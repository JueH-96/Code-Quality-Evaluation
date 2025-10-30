import sys
import math

def get_pi(s):
    """
    Computes the Knuth-Morris-Pratt (KMP) prefix function (pi table).
    pi[i] is the length of the longest proper prefix of s[0...i]
    that is also a suffix of s[0...i].
    This is used to find the length of the smallest repeating unit of a string.
    """
    n = len(s)
    if n == 0:
        return []
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi

def solve():
    """
    Solves a single test case.
    """
    S = sys.stdin.readline().strip()
    X = sys.stdin.readline().strip()
    Y = sys.stdin.readline().strip()
    
    ls = len(S)
    c0_X = X.count('0')
    c1_X = len(X) - c0_X
    c0_Y = Y.count('0')
    c1_Y = len(Y) - c0_Y

    # Length equality implies: (c0_X - c0_Y)*ls = (c1_Y - c1_X)*lt
    d0 = c0_X - c0_Y
    d1 = c1_Y - c1_X

    # Case 1: The number of S-blocks is the same.
    if d0 == 0:
        # If c0_X = c0_Y, a solution always exists.
        # Either T is forced to be empty (if c1_X != c1_Y), and the equation holds.
        # Or T can have any length (if c1_X == c1_Y), and we can pick a T that commutes with S.
        sys.stdout.write("Yes
")
        return

    # Case 2: The number of S-blocks is different (d0 != 0).
    # d0*ls = d1*lt
    
    # If d1 is 0, d0*ls = 0 => ls = 0 (since d0 != 0), a contradiction.
    if d1 == 0:
        sys.stdout.write("No
")
        return
        
    # For ls > 0, lt >= 0, d0 and d1 must have the same sign.
    if d0 * d1 <= 0:
        sys.stdout.write("No
")
        return

    # Let S = u^k_S. We seek T = u^k_T.
    # d0 * k_S = d1 * k_T => k_T = (d0 * k_S) / d1.
    # This requires (d0 * k_S) to be divisible by d1.
    
    pi = get_pi(S)
    len_of_min_period_candidate = ls - pi[ls - 1]
    
    p_S = 0
    if ls % len_of_min_period_candidate == 0:
        p_S = len_of_min_period_candidate
    else:
        p_S = ls
    k_S = ls // p_S

    g = math.gcd(d0, d1)
    d1_reduced_abs = abs(d1 // g)

    if k_S % d1_reduced_abs == 0:
        sys.stdout.write("Yes
")
    else:
        sys.stdout.write("No
")

def main():
    try:
        t_str = sys.stdin.readline()
        if not t_str: return
        t = int(t_str)
        for _ in range(t):
            solve()
    except (IOError, ValueError):
        return

if __name__ == "__main__":
    main()