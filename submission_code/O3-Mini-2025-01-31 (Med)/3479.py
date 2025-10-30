import math

class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        # Count substrings with no zeros
        count_no_zero = 0
        i = 0
        while i < n:
            if s[i] == '1':
                j = i
                # find contiguous block of ones
                while j < n and s[j] == '1':
                    j += 1
                L = j - i
                count_no_zero += L * (L + 1) // 2
                i = j
            else:
                i += 1

        # Collect positions of zeros
        pos = [i for i, ch in enumerate(s) if ch == '0']
        m = len(pos)
        # No substring with zeros qualifies if there are none
        if m == 0:
            return count_no_zero

        # T: maximum zeros count that could possibly satisfy L >= z^2+z.
        # We need z such that z^2+z <= n.
        T = int((-1 + math.sqrt(1 + 4 * n)) // 2)
        # But also, z cannot exceed the total number of zeros in s.
        T = min(T, m)
        
        ans = count_no_zero

        # Function to count number of pairs (u,v) with
        #  u in [0, A-1], v in [0, B-1], such that u+v >= r.
        def count_pairs(A, B, r):
            # maximum possible sum is (A-1)+(B-1) = A+B-2.
            if r <= 0:
                return A * B
            if r > A + B - 2:
                return 0
            # We want S = sum_{u=0}^{A-1} min(B, max(0, r - u)), for u where r-u >0.
            # We will compute this sum in O(1) using a simple break-up.
            U = min(A, r)  # u from 0 to U-1 have r-u positive.
            # Let T_u = max(0, r - B). Then for u < T_u the term is B; for u>= T_u term is r - u.
            u_break = max(0, r - B)
            if u_break > U:
                u_break = U
            # Sum for u in [0, u_break-1]: each term = B.
            part1 = u_break * B
            # For u = u_break to U-1, term = r - u.
            count_terms = U - u_break
            # Sum of (r - u) for u = u_break to U-1 equals:
            # = count_terms * r - sum_{u=u_break}^{U-1} u
            # And sum_{u=u_break}^{U-1} = (U-1 + u_break)*count_terms // 2.
            part2 = count_terms * r - ( (U - 1 + u_break) * count_terms ) // 2
            S = part1 + part2
            return A * B - S

        # For each possible z >= 1 (up to T), count substrings that contain exactly z zeros and satisfy condition.
        # Each such substring is identified by taking a consecutive block of z zeros from pos[].
        for z in range(1, T + 1):
            # iterate over all consecutive groups of z zeros
            for i in range(0, m - z + 1):
                # left boundary: the substring must start no earlier than L_min.
                if i == 0:
                    L_min = 0
                else:
                    L_min = pos[i - 1] + 1
                # And must start at or before pos[i] (to include pos[i] in the substring)
                A = pos[i] - L_min + 1  # number of choices for the left endpoint

                # right boundary: the substring must end no later than R_max.
                if i + z < m:
                    R_max = pos[i + z] - 1
                else:
                    R_max = n - 1
                B = R_max - pos[i + z - 1] + 1  # choices for right endpoint

                # The substring already contains the block from pos[i] to pos[i+z-1]. Its "base length" is:
                base = pos[i + z - 1] - pos[i] + 1
                # The additional length added by left and right (via u and v) is: u+v.
                # The condition L_total = base + u+v >= z^2 + z yields u+v >= (z^2+z - base).
                need = z * z + z - base

                # Count the number of (u,v) pairs (with u in [0, A-1] and v in [0, B-1]) with u+v >= need.
                valid_pairs = count_pairs(A, B, need)
                ans += valid_pairs
        
        return ans

# For testing purposes:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    s1 = "00011"
    print(sol.numberOfSubstrings(s1))  # Expected output: 5
    # Example 2:
    s2 = "101101"
    print(sol.numberOfSubstrings(s2))  # Expected output: 16