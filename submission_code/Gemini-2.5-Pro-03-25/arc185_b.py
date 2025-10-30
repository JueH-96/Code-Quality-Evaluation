# YOUR CODE HERE
import sys

def solve():
    """
    Solves a single test case. Reads input N and sequence A, determines if A
    can be transformed into a non-decreasing sequence using the allowed operation,
    and prints "Yes" or "No".
    """
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Compute prefix sums of the sequence A.
    # S[k] will store the sum A_1 + ... + A_k (using 1-based indexing for concept).
    # In the code, S[k] stores sum A[0]...A[k-1]. So S[k] is the sum of the first k elements.
    S = [0] * (N + 1)
    for k in range(N):
        S[k+1] = S[k] + A[k] 

    # The total sum of the sequence is S[N].
    S_N = S[N]

    # The operation allows choosing i < j and performing A_i += 1, A_j -= 1.
    # This operation conserves the total sum of the sequence.
    # Any reachable sequence B must have the same total sum S_N.
    # The operation essentially transfers value from a later index j to an earlier index i.
    # This implies that the prefix sum S_k = sum_{l=1}^k A_l can only increase or stay the same for k < N.
    # Thus, for any sequence B reachable from A, its prefix sums T_k must satisfy T_k >= S_k for k=1..N-1.
    # Also, T_N = S_N because the total sum is conserved.

    # The problem asks if there exists *any* non-decreasing sequence B that can be reached from A.
    # A sequence B is reachable from A if and only if T_k >= S_k for k=1..N-1 and T_N = S_N.
    # So the problem reduces to: Does there exist a non-decreasing sequence B such that T_k >= S_k for k=1..N-1 and T_N = S_N?

    # A known result states that such a non-decreasing sequence B exists if and only if
    # the average value of the first k elements is less than or equal to the average value of the whole sequence,
    # for all k = 1, ..., N.
    # That is, the condition S_k / k <= S_N / N must hold for all k = 1, ..., N.
    
    # To avoid potential issues with floating-point arithmetic and precision, we rewrite this condition as:
    # k * S_N >= N * S_k
    # This check must hold for k = 1, ..., N.
    
    # We perform this check using integer arithmetic. Python's integers handle arbitrarily large values,
    # so we don't need to worry about overflow for the intermediate products like k * S_N and N * S[k].
    
    possible = True
    
    # Iterate k from 1 to N. The condition must hold for all k in this range.
    for k in range(1, N + 1): 
         # S[k] contains the sum of the first k elements A[0]...A[k-1].
         # Check the condition k * S_N >= N * S[k].
         if k * S_N < N * S[k]:
             # If the condition fails for any k, it's impossible.
             possible = False
             break

    # Output the result based on whether the condition held for all k.
    if possible:
        print("Yes")
    else:
        print("No")

# Read the number of test cases from standard input.
T = int(sys.stdin.readline())
# Process each test case.
for _ in range(T):
    solve()