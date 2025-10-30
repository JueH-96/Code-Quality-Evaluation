import sys

def solve():
    """
    Calculates the maximum possible height of the head of the topmost giant.
    The height of the topmost giant P_N from the ground is determined by the
    height of the shoulders of the giant below it (P_{N-1}) plus the head height
    of P_N relative to its support point (P_{N-1}'s shoulders).

    Let h_S(P_i) be the height of giant P_i's shoulders from the ground, and
    h_H(P_i) be the height of giant P_i's head from the ground.

    P_1 is on the ground (height 0):
    h_S(P_1) = A_{P_1}
    h_H(P_1) = B_{P_1}

    For i = 1, ..., N-1, P_{i+1} is on P_i's shoulders at height h_S(P_i):
    h_S(P_{i+1}) = h_S(P_i) + A_{P_{i+1}}
    h_H(P_{i+1}) = h_S(P_i) + B_{P_{i+1}}

    We want to maximize h_H(P_N).
    h_H(P_N) = h_S(P_{N-1}) + B_{P_N}

    Expanding h_S(P_{N-1}):
    h_S(P_{N-1}) = h_S(P_{N-2}) + A_{P_{N-1}}
                 = (h_S(P_{N-3}) + A_{P_{N-2}}) + A_{P_{N-1}}
                 = ...
                 = h_S(P_1) + A_{P_2} + ... + A_{P_{N-1}}
                 = A_{P_1} + A_{P_2} + ... + A_{P_{N-1}}
                 = \sum_{j=1}^{N-1} A_{P_j}

    So, h_H(P_N) = \sum_{j=1}^{N-1} A_{P_j} + B_{P_N}.

    The set of giants {P_1, P_2, ..., P_{N-1}} is the set of all giants {1, 2, ..., N}
    excluding the topmost giant P_N. Let k be the index of the giant P_N (P_N = k).
    Then {P_1, P_2, ..., P_{N-1}} = {1, 2, ..., N} \ {k}.
    \sum_{j=1}^{N-1} A_{P_j} = \sum_{i \in \{1, ..., N\} \setminus \{k\}} A_i
                            = (\sum_{i=1}^N A_i) - A_k.

    Substituting this back into the formula for h_H(P_N):
    h_H(P_N) = (\sum_{i=1}^N A_i) - A_k + B_k
             = (\sum_{i=1}^N A_i) + (B_k - A_k).

    The sum \sum_{i=1}^N A_i is constant regardless of the permutation.
    To maximize h_H(P_N), we must choose the giant P_N (whose index is k)
    such that the term (B_k - A_k) is maximized.

    The maximum possible height is (\sum_{i=1}^N A_i) + \max_{i \in \{1, ..., N\}} (B_i - A_i).
    """
    
    # Read the number of giants
    N = int(sys.stdin.readline())
    
    sum_a = 0
    # Initialize max_b_minus_a. Since A_i <= B_i, B_i - A_i >= 0.
    # Initializing to 0 is safe.
    max_b_minus_a = 0 
    
    # Read the A_i and B_i values for all giants
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        
        # Add A_i to the total sum of A values
        sum_a += a
        
        # Calculate the difference B_i - A_i
        diff = b - a
        
        # Update the maximum difference found so far
        max_b_minus_a = max(max_b_minus_a, diff)
    
    # The maximum possible head height is the total sum of all A_i plus the maximum (B_i - A_i)
    max_head_height = sum_a + max_b_minus_a
    
    # Print the result
    print(max_head_height)

# Execute the solve function
solve()