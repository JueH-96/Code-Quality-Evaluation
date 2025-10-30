import sys
import math

def solve():
    """
    Solves the problem by checking an invariant based on the greatest common divisor (GCD).
    """
    try:
        # Fast I/O
        input = sys.stdin.readline
        
        N_str, X_str, Y_str = input().split()
        N = int(N_str)
        X = int(X_str)
        Y = int(Y_str)
        
        S = input().strip()
        T = input().strip()
    except (IOError, ValueError):
        # Handle potential empty lines or invalid format at the end of input
        return

    # A basic necessary condition is that the total number of '1's must be the same.
    # This check is implicitly covered by the main logic but can be a fast exit.
    if S.count('1') != T.count('1'):
        print("No")
        return

    # The core insight is that the operations preserve the residue class of a '1's index modulo g = gcd(X, Y).
    # An operation moves a '1' from index j to j +/- X. Since g divides X, j +/- X is congruent to j (mod g).
    # Therefore, the number of '1's in S and T for each residue class modulo g must be identical.
    
    g = math.gcd(X, Y)

    # We check this condition for each residue class k from 0 to g-1.
    for k in range(g):
        s_ones_in_subsequence = 0
        t_ones_in_subsequence = 0
        
        # Iterate through the subsequence corresponding to residue class k
        for i in range(k, N, g):
            if S[i] == '1':
                s_ones_in_subsequence += 1
            if T[i] == '1':
                t_ones_in_subsequence += 1
        
        # If the counts of '1's differ for any subsequence, it's impossible.
        if s_ones_in_subsequence != t_ones_in_subsequence:
            print("No")
            return
            
    # If the counts match for all g subsequences, it's possible.
    print("Yes")

solve()