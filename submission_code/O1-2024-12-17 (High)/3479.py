class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        """
        We need to count the number of substrings whose number of '1's is >= (number of '0's)^2.
        In particular, such substrings can have at most z zeros where z^2 + z <= len(s).
        For length up to 4e4, the largest z that can satisfy z^2+z <= 40000 is at most 199.
        
        Key observations / approach:
        
        1) Any substring with more than ~199 zeros cannot satisfy #ones >= (#zeros)^2,
           because we'd need substring length >= z^2 + z, which would exceed the entire string length.
           
        2) We split the problem into:
           - Substrings that contain no zeros at all (z=0).
             Those automatically satisfy ones >= 0^2. We can count these by summing over
             blocks of consecutive '1's.
           - Substrings that contain at least one zero, but at most 199 zeros,
             and satisfy #ones >= (#zeros)^2. We handle these by grouping zeros in s and using
             a combinatorial counting approach per block of zeros.
             
        Steps in detail:
        
        Let n = len(s).
        - Collect all positions of '0' in an array zeroPos (length M).
        - If M=0, then s has only '1's, so all substrings are valid. Return n*(n+1)//2.
        
        - Otherwise, first count all substrings that have no zeros:
          We look at gaps between consecutive zeros and also before the first zero and after the last zero.
          If a gap (block of '1's) has length b, it contributes b*(b+1)//2 zero-free substrings.
          
        - Then, let Z = [-1] + zeroPos + [n] be an extended list of zero boundaries.
          The index p runs from 1..M, and q from p.. up to p+198 (bounded by M).
          That sub-range corresponds to the substring that includes exactly the zeros with indices [p..q] in Z
          (i.e. zeroPos[p-1..q-1] in the original zeroPos array).
          
          • Number of zeros, z = q-p+1.
          • We require #ones >= z^2 for that substring.
            The substring must have length ≥ z^2 + z in total.
          • The left side of such a substring can range from L in [ Z[p-1]+1 .. Z[p] ] so as not to include
            any zero outside p..q from the left.
          • The right side can range from R in [ Z[q] .. Z[q+1]-1 ] so as not to include any zero to the right of q.
          
          We then count how many pairs (L, R) satisfy (R - L + 1) >= (z^2 + z).
          Let offset = (z^2 + z) - 1. Then we want R >= L + offset.
          
          We do a small two-pointer scan over L in [leftMin..leftMax], with R starting at R_min.
          For each L, we raise R to at least (L + offset). If that R is <= R_max, we add (R_max - R + 1) to the count.
          
        - Summing all these counts gives the final result.
        
        Complexity considerations:
        - M = number of zeros. If M = 0 or M very large, we still only consider sub-ranges up to z=199.
        - In practice, this yields an O(n * 200) worst-case complexity, which can be done carefully in Python.
        """
        
        s_arr = s
        n = len(s_arr)
        
        # Positions of zeros
        zeroPos = []
        for i, ch in enumerate(s_arr):
            if ch == '0':
                zeroPos.append(i)
        
        M = len(zeroPos)
        
        # If no zeros, all substrings are valid
        if M == 0:
            return n * (n + 1) // 2
        
        # Count substrings with no zeros (blocks of consecutive '1's)
        result = 0
        prev = -1  # index of previous zero
        for zpos in zeroPos:
            block_len = zpos - (prev + 1)
            if block_len > 0:
                result += block_len * (block_len + 1) // 2
            prev = zpos
        # Last block after the last zero
        block_len = n - (prev + 1)
        if block_len > 0:
            result += block_len * (block_len + 1) // 2
        
        # Extended zero array for boundary handling
        # Z[p] == zeroPos[p-1] when p>=1, plus we have Z[0] = -1, Z[M+1] = n
        Z = [-1] + zeroPos + [n]
        
        # Count substrings that contain from 1 up to ~199 zeros (z^2+z <= n)
        # We'll do p in 1..M, q in p..(p+198), ensuring q <= M
        # z = q-p+1, offset = z^2 + z - 1
        # left side: [Z[p-1]+1.. Z[p]]
        # right side: [Z[q].. Z[q+1] - 1]
        # must have length >= z^2+z
        
        for p in range(1, M + 1):
            left_min = Z[p - 1] + 1
            left_max = Z[p]
            
            if left_min > left_max:
                continue
            
            for q in range(p, min(M, p + 198) + 1):
                # Ensure q is within 1..M
                if q > M:
                    break
                z = q - p + 1
                offset = z * z + z - 1  # (z^2 + z) - 1
                
                r_min = Z[q]
                r_max = Z[q + 1] - 1
                if r_min > r_max:
                    continue
                
                # Check if even taking L=left_min and R=r_max can yield enough length
                # max_possible_length = (r_max - left_min + 1)
                # We need >= z^2 + z.
                if (r_max - left_min + 1) < (z * z + z):
                    continue
                
                # Two-pointer / sliding approach over L
                L_upper = r_max - offset  # if L > L_upper, threshold = L+offset > r_max => no valid R
                if L_upper < left_min:
                    # Then no L can satisfy
                    continue
                
                actual_left_max = min(left_max, L_upper)
                
                partial_count = 0
                r = r_min
                
                L = left_min
                while L <= actual_left_max:
                    threshold = L + offset
                    if r < threshold:
                        r = threshold
                    if r > r_max:
                        break
                    partial_count += (r_max - r + 1)
                    L += 1
                
                result += partial_count
        
        return result