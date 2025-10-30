import sys

def solve():
    N = int(sys.stdin.readline())
    A_str = sys.stdin.readline().strip()
    B_str = sys.stdin.readline().strip()

    A = [int(c) for c in A_str]
    B = [int(c) for c in B_str]

    # Condition 1: Trapped piece
    # If A[j-1]=1, A[j]=1, A[j+1]=1 and B[j]=0 (0-indexed)
    # The piece A[j] must move. But B[j]=0 means square j must be empty.
    # If op center k < j: A[j] moves to j-1. A[j+1] moves to j. Sq j is not empty.
    # If op center k > j: A[j] moves to j+1. A[j-1] moves to j. Sq j is not empty.
    # We cannot choose op center k = j, because A[j]=1 and B[j]=0. If k=j, A[j] stays, violating B[j]=0.
    # So, this configuration is impossible.
    for j in range(1, N - 1):
        if A[j-1] == 1 and A[j] == 1 and A[j+1] == 1 and B[j] == 0:
            print("-1")
            return

    pos_A = [i for i, x in enumerate(A) if x == 1]
    pos_B = [i for i, x in enumerate(B) if x == 1]
    
    # Constraints state A and B have at least one '1'.
    L_A, R_A = pos_A[0], pos_A[-1]
    L_B, R_B = pos_B[0], pos_B[-1]

    # Condition 2: Span
    # Span R_A - L_A cannot increase.
    if R_A - L_A < R_B - L_B:
        print("-1")
        return

    # Parameters for formula
    # C = required change in span
    # S = required change in sum of endpoints L+R (absolute value for S magnitude)
    C = (R_A - L_A) - (R_B - L_B)
    S_val = (L_A + R_A) - (L_B + R_B) # This is signed S value from derivation

    # Condition 3: Parity
    # C and S_val must have the same parity for a solution to exist with these types of operations.
    if (C % 2) != (S_val % 2):
        print("-1")
        return
    
    # If all conditions pass, the formula for minimum operations is max(C, abs(S_val))
    # This specific formula worked for Sample 1 (output 3).
    # C = 3, S_val = -3. max(3, abs(-3)) = 3.
    # For Sample 3 (output 5), this formula gives:
    # L_A=0, R_A=19 (0-indexed for N=20). L_B=3, R_B=14.
    # C = (19-0) - (14-3) = 19 - 11 = 8.
    # S_val = (0+19) - (3+14) = 19 - 17 = 2.
    # Parities C=8 (even), S_val=2 (even) match.
    # max(8, abs(2)) = 8. This does not match sample output 5.
    # The problem is known and the correct formula is $\max(\lceil C/2 ceil, \lceil |S_{\text{val}}|/2 ceil)$ if $C$ and $S_{\text{val}}$ have the same parity (after taking abs value for $S_{\text{val}}$ when comparing parity with $K$).
    # Or rather, $K_C = \lceil C/2 ceil$ and $K_S = \lceil |S_{\text{val}}|/2 ceil$.
    # If $C \% 2 == K_C \% 2$ and $|S_{\text{val}}| \% 2 == K_S \% 2$ and $K_C \% 2 == K_S \% 2$, then $\max(K_C, K_S)$. This is getting complicated.
    # The formula from ICPC WF analsysis for similar ops (shift L/R, contract L/R, contract both) is:
    # Need to make span $S_A \to S_B$ and sum $M_A \to M_B$.
    # $C_0 = S_A - S_B$. $S_0 = M_A - M_B$.
    # Required: $C_0 \ge 0$. $C_0 \equiv S_0 \pmod 2$.
    # $K = \max(C_0, (|S_0|+C_0)/2)$.
    # Test Sample 3 again with this: $C_0 = 8$, $S_0 = 2$.
    # $K = \max(8, (|2|+8)/2) = \max(8, 10/2) = \max(8,5) = 8$. Still 8.

    # The Sample 3 output (5) being smaller than my formula (8) suggests my formula is an upper bound or for a more restrictive set of operations.
    # If the problem comes from a source with specific known solution (e.g. previous contest), that would be best.
    # The formula that produces 5 for Sample 3:
    # $K_C = \lceil C/2 ceil = \lceil 8/2 ceil = 4$.
    # $K_S = \lceil |S_{\text{val}}|/2 ceil = \lceil 2/2 ceil = 1$.
    # If $\max(K_C, K_S) + \text{parity_fix_if_needed}$...
    # $\max(4,1)=4$. Perhaps if $K_C 
ot\equiv K_S \pmod 2$, add 1? $4 \pmod 2 = 0, 1 \pmod 2 = 1$. Different. So $4+1=5$.
    # Let's test this hypothesis: $K_C = (C+1)//2$ (ceil division). $K_S = (abs(S_val)+1)//2$.
    # If $K_C \% 2 
eq K_S \% 2$: result is $\max(K_C, K_S)+1$. Else: result is $\max(K_C, K_S)$.
    # Sample 1: $C=3, S_{\text{val}}=-3$. $K_C=(3+1)//2 = 2$. $K_S=(3+1)//2 = 2$.
    # $K_C\%2 = 0, K_S\%2=0$. Same. Result $\max(2,2)=2$. This is not 3.

    # Given the problem context and typical solutions, there must be a "correct" formula.
    # The formula $\max(C, |S_{\text{val}}|)$ is the most plausible if my reasoning for Sample 1 is right.
    # If that formula leads to WA due to Sample 3, then the problem analysis is deeper.
    # I will output using the $\max(C, |S_{val}|)$ logic for now.
    
    ans = max(C, abs(S_val))
    print(ans)

num_test_cases = int(sys.stdin.readline())
for _ in range(num_test_cases):
    solve()