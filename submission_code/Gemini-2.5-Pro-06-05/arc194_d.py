import sys
from collections import Counter

# It's a deep recursion problem, N can be up to 5000.
# The recursion depth can be N/2.
sys.setrecursionlimit(5005)

def solve():
    """
    Main solver function.
    """
    try:
        N = int(sys.stdin.readline())
        S = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # Handle empty input for local testing
        return

    MOD = 998244353

    # Precomputation for factorials and modular inverse factorials
    fact = [1] * (N + 1)
    inv_fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    
    inv_fact[N] = pow(fact[N], MOD - 2, MOD)
    for i in range(N - 1, -1, -1):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

    # Precomputation for matching parentheses
    match = [-1] * N
    stack = []
    for i in range(N):
        if S[i] == '(':
            stack.append(i)
        elif stack:
            j = stack.pop()
            match[j] = i

    # Memoization table for the recursive solver
    memo = {}

    # Using two hash functions to reduce collision probability
    P1, M1 = 31, 10**9 + 7
    P2, M2 = 37, 10**9 + 9
    
    # Hash values for parentheses characters, can be any distinct integers
    H_OPEN1, H_CLOSE1 = 40, 41
    H_OPEN2, H_CLOSE2 = 40, 41

    def solve_recursive(i, j):
        """
        Recursively computes the number of variations and a canonical hash for S[i:j+1].
        Returns: (number_of_ways, (hash1, hash2))
        """
        if i > j:
            # Empty string case
            # A neutral hash value for polynomial hashing (multiplicative identity)
            return 1, (1, 1)
        
        if (i, j) in memo:
            return memo[(i, j)]

        end_of_first = match[i]
        
        # Case 1: S[i:j+1] is a primitive VPS, like (A)
        if end_of_first == j:
            ways, h_A = solve_recursive(i + 1, j - 1)
            
            # The number of ways for (A) is the same as for A, as it's one block.
            # The canonical form is (C(A)). We compute its hash.
            h1_A, h2_A = h_A
            
            new_h1 = (H_OPEN1 * P1 + h1_A) % M1
            new_h1 = (new_h1 * P1 + H_CLOSE1) % M1
            
            new_h2 = (H_OPEN2 * P2 + h2_A) % M2
            new_h2 = (new_h2 * P2 + H_CLOSE2) % M2
            
            res = (ways, (new_h1, new_h2))
            memo[(i, j)] = res
            return res

        # Case 2: S[i:j+1] is a composite VPS, like P1 P2 ... Pk
        # where Pi = (Ai)
        sub_problem_results = []
        curr = i
        while curr <= j:
            end = match[curr]
            # Pi is S[curr:end+1], Ai is S[curr+1:end]
            # We solve for Ai to determine the type of Pi
            sub_problem_results.append(solve_recursive(curr + 1, end - 1))
            curr = end + 1
            
        k = len(sub_problem_results)
        
        # Group components by their type. A type is defined by (ways, hash) of its inner part.
        counts = Counter(sub_problem_results)
        
        # Calculate total ways.
        # It's product of ways from subproblems, times permutation ways.
        total_ways = fact[k]
        for type_tuple, count in counts.items():
            ways_per_comp, _ = type_tuple
            total_ways = (total_ways * inv_fact[count]) % MOD
            total_ways = (total_ways * pow(ways_per_comp, count, MOD)) % MOD
            
        # Calculate canonical hash.
        # Canonical form is C(P'_1)C(P'_2)...C(P'_k) where P' are sorted.
        # The type of Pi=(Ai) is determined by C(Pi)=(C(Ai)).
        # We need to compute hashes for each C(Pi) and then sort them.
        component_hashes = []
        for type_tuple, count in counts.items():
            _, h_A = type_tuple
            h1_A, h2_A = h_A
            
            # Hash of C(Pi) = (C(Ai))
            h_comp1 = (H_OPEN1 * P1 + h1_A) % M1
            h_comp1 = (h_comp1 * P1 + H_CLOSE1) % M1
            
            h_comp2 = (H_OPEN2 * P2 + h2_A) % M2
            h_comp2 = (h_comp2 * P2 + H_CLOSE2) % M2
            
            component_hashes.extend([(h_comp1, h_comp2)] * count)

        component_hashes.sort()

        # Combine the sorted hashes using polynomial hashing.
        new_h1, new_h2 = 1, 1 # Neutral hash
        for h1, h2 in component_hashes:
            new_h1 = (new_h1 * P1 + h1) % M1
            new_h2 = (new_h2 * P2 + h2) % M2
            
        res = (total_ways, (new_h1, new_h2))
        memo[(i, j)] = res
        return res

    final_ways, _ = solve_recursive(0, N - 1)
    print(final_ways)

solve()