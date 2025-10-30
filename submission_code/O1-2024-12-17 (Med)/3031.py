class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        """
        We want p[i][j] = (product of all elements in grid except grid[i][j]) % 12345.

        Because 12345 = 3 * 5 * 823 (not a prime), we cannot use a single product and
        modular inverse (it may not exist if gcd(grid[i][j], 12345) != 1). Instead,
        we use the fact that 3, 5, and 823 are pairwise coprime and combine results
        via the Chinese Remainder Theorem (CRT).

        High-level approach:
          1) Flatten the matrix into a 1D list arr of length K = n*m (K <= 10^5).
          2) For each prime p in (3, 5, 823):
             - Count how many elements are 0 mod p (call this zero_count_p).
             - Build prefix_prod_p and suffix_prod_p, which hold products of the
               non-zero mod p elements up to/from each position.
          3) For each index i (each element in the flattened list):
             - Let r_p = product(all except arr[i]) mod p, computed by:
                 if zero_count_p > 1:
                     r_p = 0  # because at least one zero remains in the product
                 elif zero_count_p == 1:
                     if arr[i] % p == 0:
                         # removing arr[i] removes the only zero => product of all non-zero mod p
                         r_p = (prefix_prod_p[i] * suffix_prod_p[i+1]) % p
                     else:
                         # zero still in product => 0
                         r_p = 0
                 else:  # zero_count_p == 0
                     r_p = (prefix_prod_p[i] * suffix_prod_p[i+1]) % p
            - Combine (r_3, r_5, r_823) via CRT to get r mod 12345.
          4) Reshape the 1D result back to n x m.

        Time complexity: O(K), which is feasible for K <= 1e5.
        """

        import sys
        sys.setrecursionlimit(10**7)

        # Flatten the grid into a single list for easier prefix/suffix handling
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        arr = []
        for row in grid:
            arr.extend(row)
        K = len(arr)  # K = n*m

        MOD = 12345
        # The prime factors of 12345 = 3, 5, 823
        primes = [3, 5, 823]

        # For each prime, we will store:
        # zero_count[p], prefix_prod[p], suffix_prod[p]
        # We'll keep them in a dictionary keyed by p
        zero_counts = {}
        prefix_prods = {}
        suffix_prods = {}

        for p in primes:
            # Count zeros mod p
            zc = 0
            for val in arr:
                if val % p == 0:
                    zc += 1
            zero_counts[p] = zc

            # Build prefix product (ignoring zeros in multiplication)
            prefix = [0]*(K+1)
            prefix[0] = 1
            for i in range(K):
                prefix[i+1] = prefix[i]
                if arr[i] % p != 0:
                    prefix[i+1] = (prefix[i+1] * (arr[i] % p)) % p

            # Build suffix product (ignoring zeros in multiplication)
            suffix = [0]*(K+1)
            suffix[K] = 1
            for i in range(K-1, -1, -1):
                suffix[i] = suffix[i+1]
                if arr[i] % p != 0:
                    suffix[i] = (suffix[i] * (arr[i] % p)) % p

            prefix_prods[p] = prefix
            suffix_prods[p] = suffix

        # Precompute CRT constants for mod 3, 5, 823
        # M = 3 * 5 * 823 = 12345
        # Let M1 = 5*823 = 4115, M2 = 3*823 = 2469, M3 = 3*5 = 15
        # We want x such that:
        #   x % 3   = r3
        #   x % 5   = r5
        #   x % 823 = r823
        # x = sum( r_i * M_i * inv(M_i, p_i) ) mod 12345, i in {1,2,3}
        M = 12345
        p1, p2, p3 = 3, 5, 823
        M1 = 5 * 823    # product of the other two primes for p=3
        M2 = 3 * 823    # product of the other two primes for p=5
        M3 = 3 * 5      # product of the other two primes for p=823

        # Modular inverses with Python 3.8+ pow(a, -1, m)
        invM1_mod3 = pow(M1, -1, p1)   # M1 inverse mod 3
        invM2_mod5 = pow(M2, -1, p2)   # M2 inverse mod 5
        invM3_mod823 = pow(M3, -1, p3) # M3 inverse mod 823

        def crt_combine(r3, r5, r823):
            # Combine remainders r3, r5, r823 via the standard CRT formula
            c1 = (r3 * M1 * invM1_mod3)
            c2 = (r5 * M2 * invM2_mod5)
            c3 = (r823 * M3 * invM3_mod823)
            return (c1 + c2 + c3) % M

        # Compute the product-except-self mod 12345 for each element
        result_1d = [0]*K

        for i in range(K):
            # For each prime, compute r_p = product(all except arr[i]) mod p
            # depending on zero_count logic
            remainders = []
            for p in primes:
                zc = zero_counts[p]
                pref = prefix_prods[p]
                suf = suffix_prods[p]
                if zc > 1:
                    # At least one other zero remains => product is 0 mod p
                    rp = 0
                elif zc == 1:
                    # If arr[i] is the unique zero mod p => remove it and multiply the rest
                    if arr[i] % p == 0:
                        rp = (pref[i] * suf[i+1]) % p
                    else:
                        # That unique zero is still in the product
                        rp = 0
                else:
                    # zc == 0 => normal prefix*suffix
                    rp = (pref[i] * suf[i+1]) % p
                remainders.append(rp)

            # remainders = [r3, r5, r823]
            r3, r5, r823 = remainders
            val = crt_combine(r3, r5, r823)
            result_1d[i] = val

        # Reshape result_1d into n x m
        ans = []
        idx = 0
        for _ in range(n):
            row = result_1d[idx:idx+m]
            idx += m
            ans.append(row)

        return ans