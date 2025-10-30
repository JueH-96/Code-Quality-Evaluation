import collections

class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        MOD = 10**9 + 7

        # Handle n=1 specifically. The operation "0 < l < n" cannot be performed.
        # If n=1, no operations are possible.
        # So, the string can only remain unchanged.
        # It can transform to t only if s is already t and k is 0 (no operations performed).
        if n == 1:
            return 1 if s == t and k == 0 else 0
        
        # Determine the initial state: is s already t?
        is_s_equal_t = (s == t)
        
        # If s is not t, check if t is even a cyclic shift of s.
        # If t is not a cyclic shift of s, it's impossible to transform.
        # A string `t` is a cyclic shift of `s` if `t` is a substring of `s+s`.
        if not is_s_equal_t:
            if t not in (s + s):
                return 0
        
        # Calculate the smallest period length 'd' of string 's'.
        # This is crucial for determining the number of distinct cyclic shifts.
        # We use the KMP algorithm's LPS (Longest Proper Prefix which is also a Suffix) array.
        pi = [0] * n
        for i in range(1, n):
            j = pi[i-1]
            while j > 0 and s[i] != s[j]:
                j = pi[j-1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j
        
        # 'd' is n minus the length of the longest proper prefix of s that is also a suffix of s.
        # If n is a multiple of potential_d, then d is the smallest period length.
        # Otherwise, the string is aperiodic (its smallest period is n itself).
        potential_d = n - pi[n-1]
        if n % potential_d != 0:
            d = n # Aperiodic case: all n cyclic shifts are distinct
        else:
            d = potential_d # Periodic case: some cyclic shifts are identical
        
        # N_D is the number of distinct cyclic shifts of string s.
        # This is n divided by the smallest period length d.
        N_D = n // d
        
        # Define the 2x2 transition matrix T for matrix exponentiation.
        # The states for our DP are:
        # State 0: The current string is 't'.
        # State 1: The current string is a distinct cyclic shift of 's' but is not 't'.
        #
        # T[0][0]: Number of ways to transition from 't' to 't' in one operation.
        # T[0][1]: Number of ways to transition from a non-target string (State 1) to 't' (State 0).
        # T[1][0]: Number of ways to transition from 't' (State 0) to a non-target string (State 1).
        # T[1][1]: Number of ways to transition from a non-target string (State 1) to another non-target string (State 1).
        
        if d == n: # All 'n' cyclic shifts are distinct (e.g., "abcd").
            # If d=n, an operation (cyclic shift by r, 0 < r < n) always transforms a string X to a *different* string Y.
            # No operation can transform X back to X if d=n, because that would imply a period shorter than n.
            T = [[0, 1],        # T[0][0]: From target 't' to 't': 0 ways.
                                 # T[0][1]: From non-target 's_prime' to 't': 1 way. (Among n-1 distinct shifts from s_prime, exactly one leads to t).
                 [n - 1, n - 2]] # T[1][0]: From target 't' to a non-target 's_prime': n-1 ways. (All operations lead to distinct non-targets).
                                 # T[1][1]: From non-target 's_prime' to another non-target 's_double_prime': n-2 ways. (Total n-1 ops, 1 leads to 't').
        else: # 's' has a period shorter than 'n'. N_D > 1 (e.g., "abab").
            # Number of operations that shift string X back to itself (i.e., X -> X): N_D - 1 ways.
            # Number of operations that shift string X to a *specific* other distinct string Y (i.e., X -> Y, Y!=X): d ways.
            T = [[(N_D - 1),            d          ],  # T[0][0]: From 't' to 't': N_D - 1 ways.
                                                     # T[0][1]: From 's_prime' (non-t) to 't': d ways.
                 [(n - N_D), (n - 1 - d) ]]           # T[1][0]: From 't' to 's_prime' (non-t): (n-1 total ops) - (N_D-1 ops to t) = n - N_D ways.
                                                     # T[1][1]: From 's_prime' (non-t) to 's_double_prime' (non-t): (n-1 total ops) - (d ops to t) = n-1-d ways.
            
        # Matrix multiplication function for two 2x2 matrices
        def multiply(mat1, mat2, mod):
            result = [[0, 0], [0, 0]]
            for i in range(2):
                for j in range(2):
                    for l in range(2):
                        result[i][j] = (result[i][j] + mat1[i][l] * mat2[l][j]) % mod
            return result

        # Compute T^k using binary exponentiation (matrix power)
        res_matrix = [[1, 0], [0, 1]] # Initialize as identity matrix (represents T^0)
        cur_matrix = T # The base matrix for exponentiation (represents T^1)

        temp_k = k
        while temp_k > 0:
            if temp_k % 2 == 1: # If the current bit of k is 1, multiply res_matrix by cur_matrix
                res_matrix = multiply(res_matrix, cur_matrix, MOD)
            cur_matrix = multiply(cur_matrix, cur_matrix, MOD) # Square cur_matrix for the next iteration (T^x -> T^2x)
            temp_k //= 2
        
        # After k operations, the `res_matrix` holds `T^k`.
        # We need the number of ways to be in State 0 ('t').
        # The initial state at k=0 determines which element of T^k we need:
        # If s started as t (State 0), we want the ways to get from State 0 to State 0. This is res_matrix[0][0].
        # If s started as not t (State 1), we want the ways to get from State 1 to State 0. This is res_matrix[0][1].
        
        if is_s_equal_t:
            return res_matrix[0][0]
        else:
            return res_matrix[0][1]