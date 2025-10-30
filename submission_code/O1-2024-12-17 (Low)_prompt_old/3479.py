class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        """
        We say a binary substring has 'dominant ones' if:
            number_of_ones >= (number_of_zeros)^2.

        We wish to count how many substrings of s have dominant ones.

        ---------------------------
        APPROACH (Prefix Zeros + Counting):
        
        Let n = len(s).

        1) Build a prefix array P of length n+1, where
               P[k] = number of '0's in s[:k]  (that is, up to but not including index k).
           Hence, for a substring s[i..j-1], the number of zeros is:
               zeros = P[j] - P[i].
        
        2) If that substring has zeros = z, then the number of ones in it is
               ones = (j - i) - z.
           The condition ones >= z^2 becomes
               (j - i) - z >= z^2
               j - i >= z*(z + 1).

        3) Rewrite j - i >= z*(z + 1)  =>  i <= j - z*(z + 1).
           Also from zeros = z =>  P[j] - P[i] = z  =>  P[i] = P[j] - z.

           So for each j and each valid z, the starts i that form a valid substring
           are precisely those i for which:
               a) P[i] = P[j] - z
               b) i <= j - (z^2 + z).

        4) Observe that z cannot be arbitrarily large. If z^2 > length of the entire string,
           it's impossible to satisfy #ones >= z^2. In practice, since len(s) <= 40000,
           z up to about 200 suffices (because 200^2 = 40000).

        5) Implementation details:
           - Compute prefix array P.
           - Gather positions of each prefix-sum value v in a list pos[v].
           - For j from 0..n:
               let v = P[j]
               for z in [0 .. min(200, v)]:
                   need_index = j - (z^2 + z)
                   if need_index < 0: break
                   # We require i in pos[v - z] such that i <= need_index
                   # The count of such i is the number of elements in pos[v - z] <= need_index.
                   total += number_of_i_in_pos[v - z]  with i <= need_index
           
           This yields the total count of valid substrings.

        Complexity:
           - Building P and pos array takes O(n).
           - For each j (there are n+1 such j), we iterate z up to 200 (worst-case 201 iterations),
             and do a binary search to count how many positions i <= need_index. Each binary search
             is O(log n). Overall ~ O(n * 200 * log n), which is feasible for n = 40000 with a
             reasonably efficient implementation.

        We'll verify this works with the given examples.
        """

        import bisect

        n = len(s)
        # 1) Build prefix array of zero-count
        #    P[k] = number of '0's in s[:k]
        P = [0] * (n + 1)
        for i in range(n):
            P[i + 1] = P[i] + (1 if s[i] == '0' else 0)

        # 2) Collect positions of each prefix-sum value in pos[v].
        #    v ranges from 0 up to n (worst-case all zeros).
        pos = [[] for _ in range(n + 1)]
        for i in range(n + 1):
            prefix_val = P[i]
            pos[prefix_val].append(i)

        total = 0

        # 3) For each j, consider all z up to min(200, P[j]).
        #    Then count how many i in pos[P[j] - z] are <= j - (z^2 + z).
        for j in range(n + 1):
            zeros_j = P[j]  # number of zeros up to index j
            # z cannot exceed zeros_j because we need P[i] = zeros_j - z >= 0
            upper_z = min(zeros_j, 200)
            for z in range(upper_z + 1):
                need_index = j - (z*z + z)
                if need_index < 0:
                    break
                target_val = zeros_j - z  # we want P[i] = target_val
                # We'll binary-search in pos[target_val] to find how many i <= need_index.
                arr = pos[target_val]
                # bisect_right gives count of elements <= need_index
                # (since it returns insertion point to the right of need_index).
                count_i = bisect.bisect_right(arr, need_index)
                total += count_i

        return total