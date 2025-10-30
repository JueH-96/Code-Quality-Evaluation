import sys
import math

# Use two large prime moduli and two bases for robustness
MOD1 = 10**9 + 7
MOD2 = 10**9 + 9
BASE1 = 31
BASE2 = 37

# Precompute powers of bases
# Max sum of |S|, |X|, |Y| over all test cases is 5 * 10^5
MAX_TOTAL_LEN = 5 * 10**5
pow1 = [1] * (MAX_TOTAL_LEN + 1)
pow2 = [1] * (MAX_TOTAL_LEN + 1)
for i in range(1, MAX_TOTAL_LEN + 1):
    pow1[i] = (pow1[i-1] * BASE1) % MOD1
    pow2[i] = (pow2[i-1] * BASE2) % MOD2

# Precompute prefix hashes for S
def build_prefix_hashes(s):
    n = len(s)
    h1 = [0] * (n + 1)
    h2 = [0] * (n + 1)
    for i in range(n):
        h1[i+1] = (h1[i] * BASE1 + (ord(s[i]) - ord('a') + 1)) % MOD1
        h2[i+1] = (h2[i] * BASE2 + (ord(s[i]) - ord('a') + 1)) % MOD2
    return h1, h2

# Get hash of substring s[l:r] (0-indexed, end exclusive)
def get_hash(h1, h2, l, r, length):
    hash1 = (h1[r] - (h1[l] * pow1[length]) % MOD1 + MOD1) % MOD1
    hash2 = (h2[r] - (h2[l] * pow2[length]) % MOD2 + MOD2) % MOD2
    return hash1, hash2

def solve():
    S = sys.stdin.readline().strip()
    X = sys.stdin.readline().strip()
    Y = sys.stdin.readline().strip()

    n0X = X.count('0')
    n1X = X.count('1')
    n0Y = Y.count('0')
    n1Y = Y.count('1')

    len_S = len(S)

    # Equation: n0X*|S| + n1X*|T| = n0Y*|S| + n1Y*|T|
    # (n1X - n1Y)*|T| = (n0Y - n0X)*|S|
    delta_n1 = n1X - n1Y
    delta_n0 = n0Y - n0X

    if delta_n1 == 0:
        if delta_n0 == 0:
            print("Yes")
        else:
            print("No")
        return

    # delta_n1 != 0
    num = delta_n0 * len_S
    den = delta_n1

    if num % den != 0:
        print("No")
        return

    len_T_req = num // den

    if len_T_req < 0:
        print("No")
        return

    if len_T_req == 0:
        # If len_T_req = 0, then num = 0. Since len_S >= 1, delta_n0 = 0.
        # The original equation becomes n0X|S| = n0Y|S|, which is true if n0X=n0Y (delta_n0=0).
        # So T empty works iff delta_n0 = 0. The calculation of len_T_req already confirmed delta_n0=0.
        print("Yes")
        return

    # Required |T| = len_T_req > 0
    h1_S, h2_S = build_prefix_hashes(S)

    # Constraints on T base string
    # Use a list/array for the base, size is gcd_period.
    t_period_gcd = len_T_req
    # Use a dummy character (e.g., ASCII 0) to indicate not defined
    t_base_chars = [0] * t_period_gcd

    p_X, o_X = 0, 0
    p_Y, o_Y = 0, 0

    while p_X < len(X) or o_X > 0 or p_Y < len(Y) or o_Y > 0:
        # Determine current blocks and their lengths
        if o_X == 0: # Start of a new block in X stream
            if p_X == len(X): # X stream exhausted, but Y stream still has content
                 print("No") # Mismatch in total length
                 return
            char_X = X[p_X]
            len_BX = len_S if char_X == '0' else len_T_req
        else: # Continuing previous block in X stream
            char_X = X[p_X - 1] # Character from X that determined this block
            len_BX = len_S if char_X == '0' else len_T_req

        if o_Y == 0: # Start of a new block in Y stream
            if p_Y == len(Y): # Y stream exhausted, but X stream still has content
                print("No") # Mismatch in total length
                return
            char_Y = Y[p_Y]
            len_BY = len_S if char_Y == '0' else len_T_req
        else: # Continuing previous block in Y stream
            char_Y = Y[p_Y - 1] # Character from Y that determined this block
            len_BY = len_S if char_Y == '0' else len_T_req

        # Calculate remaining length in current blocks
        rem_X = len_BX - o_X
        rem_Y = len_BY - o_Y

        # Length to compare in this step
        comp_len = min(rem_X, rem_Y)

        # Compare substrings
        if char_X == '0' and char_Y == '0':
            # Compare S[o_X : o_X+comp_len] and S[o_Y : o_Y+comp_len]
            if get_hash(h1_S, h2_S, o_X, o_X + comp_len, comp_len) != get_hash(h1_S, h2_S, o_Y, o_Y + comp_len, comp_len):
                print("No")
                return

        elif char_X == '0' and char_Y == '1':
            # S[o_X : o_X+comp_len] must equal T[o_Y : o_Y+comp_len]
            # Impose constraints on T base string (modulo current gcd_period)
            for k in range(comp_len):
                t_idx_in_base = (o_Y + k) % t_period_gcd
                char_S_val = ord(S[o_X + k]) - ord('a') + 1
                if t_base_chars[t_idx_in_base] != 0 and t_base_chars[t_idx_in_base] != char_S_val:
                    print("No")
                    return
                t_base_chars[t_idx_in_base] = char_S_val

        elif char_X == '1' and char_Y == '0':
            # T[o_X : o_X+comp_len] must equal S[o_Y : o_Y+comp_len]
            # Impose constraints on T base string (modulo current gcd_period)
            for k in range(comp_len):
                t_idx_in_base = (o_X + k) % t_period_gcd
                char_S_val = ord(S[o_Y + k]) - ord('a') + 1
                if t_base_chars[t_idx_in_base] != 0 and t_base_chars[t_idx_in_base] != char_S_val:
                    print("No")
                    return
                t_base_chars[t_idx_in_base] = char_S_val

        elif char_X == '1' and char_Y == '1':
            # T[o_X : o_X+comp_len] must equal T[o_Y : o_Y+comp_len]
            if o_X != o_Y:
                diff = abs(o_X - o_Y)
                new_gcd = math.gcd(t_period_gcd, diff)
                if new_gcd < t_period_gcd:
                    # Update t_period_gcd and remap existing constraints
                    new_t_base_chars = [0] * new_gcd
                    for i in range(t_period_gcd):
                        if t_base_chars[i] != 0:
                            new_idx_in_base = i % new_gcd
                            if new_t_base_chars[new_idx_in_base] != 0 and new_t_base_chars[new_idx_in_base] != t_base_chars[i]:
                                print("No")
                                return
                            new_t_base_chars[new_idx_in_base] = t_base_chars[i]
                    t_base_chars = new_t_base_chars
                    t_period_gcd = new_gcd

            # Check consistency for T=T comparison based on updated period
            for k in range(comp_len):
                idx1 = (o_X + k) % t_period_gcd
                idx2 = (o_Y + k) % t_period_gcd
                
                val1 = t_base_chars[idx1]
                val2 = t_base_chars[idx2]

                if val1 != 0 and val2 != 0 and val1 != val2:
                    print("No")
                    return
                if val1 != 0 and val2 == 0:
                    t_base_chars[idx2] = val1
                elif val1 == 0 and val2 != 0:
                    t_base_chars[idx1] = val2


        # Advance offsets and block pointers
        o_X += comp_len
        o_Y += comp_len

        if o_X == len_BX:
            o_X = 0
            p_X += 1
        if o_Y == len_BY:
            o_Y = 0
            p_Y += 1

    # Simulation finished. Both streams must have reached the end exactly.
    # The while loop condition ensures this. If we exit the loop, it means p_X == len(X), o_X == 0, p_Y == len(Y), o_Y == 0.
    print("Yes")


# Read the number of test cases
t = int(sys.stdin.readline())
for _ in range(t):
    solve()