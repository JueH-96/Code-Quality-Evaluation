import sys

def sum_pow(exp):
    return sum(F ** exp for F in range(1, 10))

def count_le_with_max_digit(T_str, max_dig):
    M = len(T_str)
    ans = 0
    tight = True
    for pos in range(M):
        if tight:
            up = int(T_str[pos])
            num_d_less = (min(up - 1, max_dig) + 1) if up > 0 else 0
            ans += num_d_less * (max_dig + 1) ** (M - pos - 1)
            if up <= max_dig:
                tight = True  # keep tight
            else:
                tight = False
                break  # no more for this path
        else:
            # not tight
            rem_len = M - pos
            ans += (max_dig + 1) ** rem_len
            break  # exit loop
    if tight:
        ans += 1
    return ans

def count_up_to(S_str):
    len_S = len(S_str)
    if len_S < 2:
        return 0
    ans = 0
    # add Snake with fewer digits, D from 2 to len_S - 1
    if len_S > 2:
        for D in range(2, len_S):
            exp = D - 1
            ans += sum_pow(exp)
    # now for exactly len_S digits
    S1 = int(S_str[0])
    # sum over F < S1
    for F in range(1, S1):
        ans += F ** (len_S - 1)
    # for F == S1
    max_dig_Y = S1 - 1
    S_rem = S_str[1:]
    cnt = count_le_with_max_digit(S_rem, max_dig_Y)
    ans += cnt
    return ans

# Read input and compute
data = sys.stdin.read().strip()
L, R = map(int, data.split())
count_R_val = count_up_to(str(R))
count_L_minus_1_val = count_up_to(str(L - 1))
answer = count_R_val - count_L_minus_1_val
print(answer)