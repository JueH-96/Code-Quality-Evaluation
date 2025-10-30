class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        """
        We want p[i][j] = (product of all entries of grid except grid[i][j]) mod 12345.
        
        Since 12345 = 3 * 5 * 823, a direct "divide in modular arithmetic" can fail if an element
        shares a common divisor with 12345. Also, a single global product mod 12345 can be zero 
        even if some partial products of size (n*m - 1) are non-zero.

        A robust approach uses prefix-suffix products for each prime factor's modulus (3, 5, 823)
        and then combines those partial products with the Chinese Remainder Theorem (CRT). 

        Steps:
          1) Flatten the matrix into a 1D array arr of length n*m for convenience.
          2) For each prime in (3, 5, 823), build:
             prefix[p][i]: product of arr[:i] mod p
             suffix[p][i]: product of arr[i:] mod p
             (We actually store prefix[p][i] in prefix[p][i+1] = product of first i arr elements,
              similarly for suffix.)
          3) For each index i, "product_except_i" mod p = prefix[p][i] * suffix[p][i+1] mod p.
          4) Combine results for mod 3, mod 5, and mod 823 via CRT to get the value mod 12345.
          5) Reshape the 1D result back into an n x m 2D matrix.

        Complexity is O(n*m), which is feasible for n*m <= 10^5.

        Example checks are consistent with the described method.
        """

        import sys
        sys.setrecursionlimit(10**7)
        
        # If the grid is empty or trivial, just return an empty result (though constraints say min 2 elements).
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        if n == 0 or m == 0: 
            return []
        
        # Flatten the matrix
        arr = []
        for row in grid:
            arr.extend(row)
        length = len(arr)  # n*m
        
        # Precompute arr mod each prime for efficiency
        MODS = [3, 5, 823]
        arr_mod = {
            p: [x % p for x in arr]
            for p in MODS
        }
        
        # Build prefix and suffix arrays for each prime p
        # We'll store them in dicts: prefix[p], suffix[p], each of length length+1
        # prefix[p][i] = product(arr[0], arr[1], ..., arr[i-1]) mod p
        # suffix[p][i] = product(arr[i], arr[i+1], ..., arr[length-1]) mod p
        prefix = {p: [0]*(length+1) for p in MODS}
        suffix = {p: [0]*(length+1) for p in MODS}
        
        for p in MODS:
            prefix[p][0] = 1
            for i in range(length):
                prefix[p][i+1] = (prefix[p][i] * arr_mod[p][i]) % p
            
            suffix[p][length] = 1
            for i in range(length-1, -1, -1):
                suffix[p][i] = (suffix[p][i+1] * arr_mod[p][i]) % p
        
        # Extended Euclidean for modular inverse
        def extended_gcd(a, b):
            """
            Returns (g, x, y) such that a*x + b*y = g = gcd(a,b).
            """
            if b == 0:
                return (a, 1, 0)
            g2, x2, y2 = extended_gcd(b, a % b)
            return (g2, y2, x2 - (a // b) * y2)
        
        def mod_inv(a, m):
            """
            Compute inverse of a mod m (assuming gcd(a,m)=1).
            """
            g, x, _ = extended_gcd(a, m)
            if g != 1:
                # No inverse if gcd != 1, but problem setup ensures usage only when gcd=1
                return None
            return x % m
        
        # We apply CRT to combine results mod (3, 5, 823) => result mod 12345
        # We'll do the standard multiple-moduli CRT approach in O(1) for each triple.
        # N = 3*5*823 = 12345
        # Suppose we have remainders r3, r5, r823 for mod 3, 5, 823.
        # Let M1=12345/3=4115, M2=12345/5=2469, M3=12345/823=15.
        # Then x = sum( r_i * M_i * inv(M_i, mod_i ) ) mod 12345.
        N = 12345
        moduli = [3, 5, 823]
        # Precompute those M_i
        Mvals = [N // p for p in moduli]
        # Precompute inverses of Mvals[i] mod moduli[i]
        invM = []
        for i, p in enumerate(moduli):
            Mi = Mvals[i]
            invM.append(mod_inv(Mi % p, p))  # inverse of Mi mod p
        
        def crt(r3, r5, r823):
            # Use the precomputed Ms and inverses
            # x = sum( r[i] * Mvals[i] * invM[i] ) mod 12345
            # mapped in order r3-> index0, r5->index1, r823->index2
            r = [r3, r5, r823]
            x = 0
            for i, p in enumerate(moduli):
                x += r[i] * Mvals[i] * invM[i]
            return x % N
        
        # Build the result array (flattened)
        p_flat = [0]*length
        
        for i in range(length):
            # product_except_i mod p = prefix[p][i] * suffix[p][i+1] mod p
            r_vals = []
            for p in MODS:
                val = (prefix[p][i] * suffix[p][i+1]) % p
                r_vals.append(val)
            # r_vals = [r_mod3, r_mod5, r_mod823]
            p_flat[i] = crt(r_vals[0], r_vals[1], r_vals[2])
        
        # Reshape p_flat into an n x m matrix
        ans = []
        index = 0
        for _ in range(n):
            row = p_flat[index : index + m]
            index += m
            ans.append(row)
        
        return ans