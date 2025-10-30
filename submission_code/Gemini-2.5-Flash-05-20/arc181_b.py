import sys

# Constants for polynomial hashing
# Using two large prime moduli to reduce collision probability
MOD1 = 10**9 + 7
MOD2 = 10**9 + 9
BASE1 = 31 # a prime > 26 (alphabet size)
BASE2 = 37 # another prime

# Precompute powers of bases up to max string length + 1
# Max length is 5*10^5, so 5*10^5 + 1 elements are needed.
MAX_LEN = 5 * 10**5
POW1 = [1] * (MAX_LEN + 1)
POW2 = [1] * (MAX_LEN + 1)
for i in range(1, MAX_LEN + 1):
    POW1[i] = (POW1[i-1] * BASE1) % MOD1
    POW2[i] = (POW2[i-1] * BASE2) % MOD2

class StringHasher:
    def __init__(self, s):
        self.n = len(s)
        self.s = s
        self.pref_hash1 = [0] * (self.n + 1)
        self.pref_hash2 = [0] * (self.n + 1)

        # Precompute prefix hashes
        for i in range(self.n):
            self.pref_hash1[i+1] = (self.pref_hash1[i] * BASE1 + (ord(s[i]) - ord('a') + 1)) % MOD1
            self.pref_hash2[i+1] = (self.pref_hash2[i] * BASE2 + (ord(s[i]) - ord('a') + 1)) % MOD2

    # Get hash of substring s[start:end] (0-indexed, end exclusive)
    def get_hash(self, start, end):
        length = end - start
        h1 = (self.pref_hash1[end] - (self.pref_hash1[start] * POW1[length]) % MOD1 + MOD1) % MOD1
        h2 = (self.pref_hash2[end] - (self.pref_hash2[start] * POW2[length]) % MOD2 + MOD2) % MOD2
        return (h1, h2)

    # Get hash of substring s starting at 'start' for 'length', handling cyclic wrap-around
    def get_cyclic_hash(self, start, length):
        if length == 0:
            return (0, 0)
        
        # Ensure 'start' is within [0, self.n-1]
        start = (start % self.n + self.n) % self.n
        
        if start + length <= self.n:
            # Substring does not wrap around
            return self.get_hash(start, start + length)
        
        # Substring wraps around
        # Part 1: from 'start' to the end of the string
        len1 = self.n - start
        h1_part1, h2_part1 = self.get_hash(start, self.n)

        # Part 2: from the beginning of the string, for the remaining length
        len2 = length - len1
        h1_part2, h2_part2 = self.get_hash(0, len2)

        # Combine hashes: h1_part1 is multiplied by BASE1^len2 to shift it to the left
        h1 = (h1_part1 * POW1[len2] + h1_part2) % MOD1
        h2 = (h2_part1 * POW2[len2] + h2_part2) % MOD2
        
        return (h1, h2)


def solve():
    S = sys.stdin.readline().strip()
    X = sys.stdin.readline().strip()
    Y = sys.stdin.readline().strip()

    len_S = len(S)
    
    # Count '0's and '1's in X and Y
    c0_X = X.count('0')
    c1_X = X.count('1')
    c0_Y = Y.count('0')
    c1_Y = Y.count('1')

    # Calculate differences for length equation: diff0 * |S| = diff1 * |T|
    diff0 = c0_X - c0_Y
    diff1 = c1_Y - c1_X

    L_T = 0 # Required length of T

    if diff1 == 0: # c1_X == c1_Y
        if diff0 == 0: # c0_X == c0_Y
            # Counts of '0's and '1's are identical for X and Y.
            # This implies |X| == |Y|.
            # Example: S=ab, X=01, Y=10. T=ab works. f(S,T,X)=abab, f(S,T,Y)=abab.
            # Choosing T=S always results in f(S,S,X) being S repeated |X| times
            # and f(S,S,Y) being S repeated |Y| times. Since |X|=|Y|, they are equal.
            sys.stdout.write("Yes
")
            return
        else: # c0_X != c0_Y
            # Equation becomes diff0 * |S| = 0. Since |S| > 0, this requires diff0 = 0,
            # which contradicts diff0 != 0. No solution for |T|.
            sys.stdout.write("No
")
            return
    else: # diff1 != 0 (c1_X != c1_Y)
        numerator = diff0 * len_S
        if numerator % diff1 != 0: # Check divisibility
            sys.stdout.write("No
")
            return
        
        L_T = numerator // diff1 # Calculate required length of T
        
        if L_T < 0: # |T| cannot be negative
            sys.stdout.write("No
")
            return
        
        if L_T == 0:
            # If |T|=0, then diff0 * |S| = 0, which implies diff0 = 0 (as |S|>0).
            # So c0_X = c0_Y. We are in the case c0_X=c0_Y AND c1_X!=c1_Y.
            # If T is the empty string, f(S,"",X) becomes S repeated c0_X times,
            # and f(S,"",Y) becomes S repeated c0_Y times. Since c0_X=c0_Y, these are equal.
            sys.stdout.write("Yes
")
            return

    # At this point, L_T > 0 and represents a valid required length for T.
    # We now simulate the string construction and check for consistency.
    # S_eq_T_offset: stores the constant 'C' such that T[k] = S[(k + C) % |S|]
    # Initialized to None, meaning no S-T relationship has been established yet.
    S_eq_T_offset = None 
    
    # Precompute hashes for S for efficient substring comparison
    hasher_S = StringHasher(S)

    # Pointers for traversing the conceptual f(S,T,X) and f(S,T,Y) strings
    idx_X, cur_len_X = 0, 0 # Current block index in X, and offset within that block
    idx_Y, cur_len_Y = 0, 0 # Current block index in Y, and offset within that block

    possible = True
    # Loop as long as there are blocks left in either X or Y
    while idx_X < len(X) and idx_Y < len(Y):
        char_type_X = X[idx_X]
        char_type_Y = Y[idx_Y]

        # Determine lengths of current blocks
        len_block_X = len_S if char_type_X == '0' else L_T
        len_block_Y = len_S if char_type_Y == '0' else L_T

        # Calculate how many characters can be matched in this step
        rem_X = len_block_X - cur_len_X
        rem_Y = len_block_Y - cur_len_Y
        match_len = min(rem_X, rem_Y)
        
        if char_type_X == char_type_Y:
            if char_type_X == '0': # Both are S blocks
                # Compare S[cur_len_X : cur_len_X + match_len] with S[cur_len_Y : cur_len_Y + match_len]
                hash_X_part = hasher_S.get_cyclic_hash(cur_len_X, match_len)
                hash_Y_part = hasher_S.get_cyclic_hash(cur_len_Y, match_len)
                if hash_X_part != hash_Y_part:
                    possible = False
                    break
            else: # Both are T blocks
                if S_eq_T_offset is None:
                    # If T is not constrained by S, T segments must match themselves.
                    # This is only possible if they are at the same relative offset in T.
                    if cur_len_X != cur_len_Y:
                        possible = False
                        break
                else:
                    # T is constrained by S. Compare the S segments that T would map to.
                    # T[k] = S[(k + S_eq_T_offset) % |S|]
                    s_offset_for_X = (cur_len_X + S_eq_T_offset + len_S) % len_S
                    s_offset_for_Y = (cur_len_Y + S_eq_T_offset + len_S) % len_S
                    
                    hash_X_implied_S_part = hasher_S.get_cyclic_hash(s_offset_for_X, match_len)
                    hash_Y_implied_S_part = hasher_S.get_cyclic_hash(s_offset_for_Y, match_len)
                    
                    if hash_X_implied_S_part != hash_Y_implied_S_part:
                        possible = False
                        break
        else: # Different block types (S vs T or T vs S)
            # Determine which block is S and which is T
            s_offset_current_block = 0
            t_offset_current_block = 0
            
            if char_type_X == '0': # X is S, Y is T
                s_offset_current_block = cur_len_X
                t_offset_current_block = cur_len_Y
            else: # X is T, Y is S
                s_offset_current_block = cur_len_Y
                t_offset_current_block = cur_len_X

            # Calculate the current required offset C if S[s_idx] = T[t_idx] means S[s_idx] = S[(t_idx + C) % |S|]
            current_C_candidate = (s_offset_current_block - t_offset_current_block)

            if S_eq_T_offset is None:
                S_eq_T_offset = current_C_candidate
            
            # Compare the actual S segment with the S segment that T would map to.
            # The actual S part from the S block
            hash_actual_S_part = hasher_S.get_cyclic_hash(s_offset_current_block, match_len)
            
            # The S part implied by T, given the established S_eq_T_offset
            s_mapping_for_T_start_offset = ((t_offset_current_block + S_eq_T_offset) % len_S + len_S) % len_S
            hash_T_implied_S_part = hasher_S.get_cyclic_hash(s_mapping_for_T_start_offset, match_len)
            
            if hash_actual_S_part != hash_T_implied_S_part:
                possible = False
                break

        # Advance pointers for the next iteration
        cur_len_X += match_len
        cur_len_Y += match_len

        if cur_len_X == len_block_X: # Current X block is exhausted
            idx_X += 1
            cur_len_X = 0
        
        if cur_len_Y == len_block_Y: # Current Y block is exhausted
            idx_Y += 1
            cur_len_Y = 0
            
    # After the loop, if no inconsistencies were found and both X and Y streams are fully processed, it's a "Yes".
    if possible:
        sys.stdout.write("Yes
")
    else:
        sys.stdout.write("No
")


# Read number of test cases
num_test_cases = int(sys.stdin.readline())
for _ in range(num_test_cases):
    solve()