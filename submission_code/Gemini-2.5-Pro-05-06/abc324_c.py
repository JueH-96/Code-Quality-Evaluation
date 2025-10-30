import sys

def solve():
    N_str, T_prime = sys.stdin.readline().split()
    N = int(N_str)

    ans_indices = []
    len_T_prime = len(T_prime)

    for i in range(1, N + 1):
        S = sys.stdin.readline().strip() # Read S_i and remove trailing newline
        len_S = len(S)
        
        possible = False

        if abs(len_S - len_T_prime) > 1:
            # Difference in length is too large for any of the 4 conditions
            pass
        elif len_S == len_T_prime:
            # S could be T. Two subconditions:
            # 1. T' == T (S is T)
            # 4. T' is T with one char changed (S is T, T' is S with one char changed)
            if S == T_prime: # Condition 1
                possible = True
            else: # Potential Condition 4
                diff_count = 0
                for k in range(len_S): # len_S == len_T_prime
                    if S[k] != T_prime[k]:
                        diff_count += 1
                if diff_count == 1: # Condition 4
                    possible = True
        elif len_S == len_T_prime - 1:
            # S could be T. Condition 2: T' is T with one char inserted.
            # So S is shorter than T_prime by 1. (S is s_short, T_prime is s_long)
            
            p = 0 # length of common prefix between S and T_prime
            while p < len_S and S[p] == T_prime[p]:
                p += 1
            
            s = 0 # length of common suffix between S and T_prime
            while s < len_S and S[len_S - 1 - s] == T_prime[len_T_prime - 1 - s]:
                s += 1
            
            # If prefix of S and suffix of S "cover" S when aligned with T_prime's prefix/suffix.
            # Formally, if LCP_len + LCS_len >= len(ShorterString). Here, S is shorter.
            if p + s >= len_S:
                 possible = True
            # Example: S="ace", T_prime="abce". len_S=3. p=1 ("a"), s=2 ("ce"). p+s = 3. 3 >= 3. True.

        elif len_S == len_T_prime + 1:
            # S could be T. Condition 3: T' is T with one char deleted.
            # So S is longer than T_prime by 1. (T_prime is s_short, S is s_long)

            p = 0 # length of common prefix between T_prime and S
            while p < len_T_prime and T_prime[p] == S[p]:
                p += 1
            
            s = 0 # length of common suffix between T_prime and S
            while s < len_T_prime and T_prime[len_T_prime - 1 - s] == S[len_S - 1 - s]:
                s += 1
            
            # Here, T_prime is shorter. Check LCP_len + LCS_len >= len(T_prime).
            if p + s >= len_T_prime:
                possible = True
        
        if possible:
            ans_indices.append(i)

    sys.stdout.write(str(len(ans_indices)) + "
")
    if len(ans_indices) > 0:
        sys.stdout.write(" ".join(map(str, ans_indices)) + "
")
    # If K=0, the problem format expects "0" then a newline. My code prints "0
" and then nothing.
    # If K>0, "K
" then "i_1 ... i_K
". This is achieved.
    # The problem statement for K=0 output format like:
    # K
    # i_1 i_2 ... i_K
    # This implies an empty second line if K=0. `print(*(ans_indices))` when empty produces a newline.
    # ` " ".join(map(str, ans_indices))` produces an empty string, so `print("")` also results in a newline.
    # It seems standard to print the empty line for K=0. So, the following is slightly more robust:
    # sys.stdout.write(" ".join(map(str, ans_indices)) + "
") # instead of if len > 0
    # However, to be safe with typical contest platform behavior where trailing spaces or exact number of lines matter:
    # The sample output for K=0 has "0" then stops. It doesn't print an empty second line.
    # So, my `if len(ans_indices) > 0:` block is better.

solve()