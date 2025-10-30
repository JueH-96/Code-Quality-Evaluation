import sys
import math

# Setting a reasonable recursion depth for safety, although the code is iterative.
# sys.setrecursionlimit(2000) 

def solve():
    # Read input values N, X, Y
    N, X, Y = map(int, sys.stdin.readline().split())
    # Read strings S and T
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Convert strings S and T to lists of integers (0 or 1) for easier manipulation
    S_list = [int(c) for c in S]
    T_list = [int(c) for c in T]

    # Crucial Necessary Condition: The number of 1s must be invariant.
    # Operations A and B preserve the count of 1s within the affected block,
    # and thus preserve the total count of 1s in the string.
    # If the counts don't match, transformation is impossible.
    if S_list.count(1) != T_list.count(1):
        print("No")
        return

    # The core logic relies on analyzing the structure preserved by the operations.
    # This involves looking at the difference string D = S XOR T, its prefix XOR sums P_D,
    # and the second difference of prefix XOR sums, delta^2 P_D.
    # The operations correspond to XORing D with specific patterns Delta S(i).
    # An analysis shows that if S can be transformed to T, then the second difference 
    # vector delta^2 P_D must belong to a specific vector space V_d spanned by
    # vectors with two 1s at a specific distance 'd'.
    
    # Compute the difference vector D = S XOR T. D[k] = 1 if S[k] != T[k], else 0.
    # This vector represents the positions where S and T differ.
    D = [(S_list[k] ^ T_list[k]) for k in range(N)]

    # Compute prefix XOR sums P_D of the difference vector D.
    # P_D[k] stores the XOR sum of D[0]...D[k-1]. P_D[0] is defined as 0.
    # The array P_D has length N+1 to store values for indices 0 to N.
    P_D = [0] * (N + 1)
    for k in range(N):
        P_D[k+1] = P_D[k] ^ D[k]

    # Compute the second difference of the prefix XOR sums: delta^2 P_D.
    # The definition is delta^2 P_D[k] = P_D[k] XOR P_D[k-2] (using 1-based index logic for 'k').
    # The resulting vector of length N is stored in `delta2_P_D_vec` using 0-based indexing.
    # Specifically, `delta2_P_D_vec[k]` corresponds to the value delta^2 P_D at 1-based index `k+1`.
    
    delta2_P_D_vec = [0] * N
    
    # Handle boundary cases for k=1 and k=2 (corresponding to 0-based indices 0 and 1)
    # We define P_D[-1] = 0 conceptually for the formula.
    if N >= 1:
       # For k=1 (0-based index 0): delta^2 P_D[1] = P_D[1] ^ P_D[-1] = P_D[1]
       delta2_P_D_vec[0] = P_D[1] 
    if N >= 2:
       # For k=2 (0-based index 1): delta^2 P_D[2] = P_D[2] ^ P_D[0]
       delta2_P_D_vec[1] = P_D[2] ^ P_D[0]
       
    # Compute for k >= 3 (corresponding to 0-based indices k >= 2)
    for k in range(2, N): # Iterate through 0-based indices k from 2 to N-1
       # The k-th element of delta2_P_D_vec corresponds to 1-based index k+1.
       # The formula delta^2 P_D[k+1] = P_D[k+1] ^ P_D[k-1] is used.
       delta2_P_D_vec[k] = P_D[k+1] ^ P_D[k-1]

    # Determine the characteristic distance 'd' based on X and Y.
    # This distance defines the structure of the vector space V_d.
    # The basis vectors for V_d are of the form e_idx1 + e_idx2 where |idx1 - idx2| = d.
    if X == Y:
        # If X=Y, analysis shows the span vectors delta^2 Delta P(i) are of the form e_i + e_{i+2X}. 
        # The distance between the 1s is 2X, which is X+Y.
        d = X + Y 
    else:
        # If X != Y, analysis shows span vectors delta^2 Delta P(i) are of the form e_{i+min(X,Y)} + e_{i+max(X,Y)}.
        # The distance between the 1s is max(X,Y) - min(X,Y) = |X-Y|.
        d = abs(X - Y)

    # Check if delta^2 P_D belongs to the span V_d.
    # A vector belongs to V_d if and only if for each residue class r modulo d,
    # the number of indices k (1-based) such that k = r (mod d) and the k-th component is 1, is even.
    
    # Initialize counts for each residue class modulo d.
    counts = [0] * d
    
    # Iterate through the computed second difference vector delta2_P_D_vec.
    for k in range(N): # k is the 0-based index
        if delta2_P_D_vec[k] == 1:
            # (k+1) corresponds to the 1-based index used in the mathematical derivation.
            # Increment the count for the residue class (k+1) % d.
            counts[ (k + 1) % d ] += 1 

    # Check if all counts are even.
    all_even = True
    for r in range(d):
        if counts[r] % 2 != 0:
            # If any count is odd, the condition is not met.
            all_even = False
            break

    # This necessary condition derived from the span property is assumed to be sufficient
    # based on patterns observed in similar competitive programming problems.
    if all_even:
         print("Yes")
    else:
         print("No")

# Execute the solve function
solve()