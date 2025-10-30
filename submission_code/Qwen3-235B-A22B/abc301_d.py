import sys
from functools import lru_cache

def main():
    S = sys.stdin.readline().strip()
    N_line = sys.stdin.readline().strip()
    N = int(N_line)
    len_S = len(S)
    
    if len_S == 0:
        print(-1)
        return
    
    bin_N = bin(N)[2:]
    len_N = len(bin_N)
    
    # Handle case where S is shorter than N's binary length
    if len_S < len_N:
        s_list = list(S)
        for i in range(len_S):
            if s_list[i] == '?':
                s_list[i] = '1'
        max_val = int(''.join(s_list), 2)
        print(max_val if max_val <= N else -1)
        return
    
    # Compute binary_N padded with leading zeros to match len_S
    binary_N = bin_N.zfill(len_S)
    
    # Compute max_T to check if it's <= N
    s_max = []
    for c in S:
        s_max.append('1' if c == '?' else c)
    max_val = int(''.join(s_max), 2)
    if max_val <= N:
        print(max_val)
        return
    
    # Proceed with DP
    @lru_cache(maxsize=None)
    def dp(i, is_tight):
        if i == len_S:
            return 0
        res = -1
        c = S[i]
        allowed_bits = []
        if c == '0':
            allowed_bits = ['0']
        elif c == '1':
            allowed_bits = ['1']
        else:
            allowed_bits = ['1', '0']  # Try '1' first
        for b in allowed_bits:
            if is_tight:
                if b > binary_N[i]:
                    continue
                new_tight = (b == binary_N[i])
            else:
                new_tight = False
            rest = dp(i + 1, new_tight)
            if rest == -1:
                continue
            bit_power = 1 << (len_S - i - 1)
            current_val = int(b) * bit_power + rest
            if current_val > res:
                res = current_val
        return res
    
    result = dp(0, True)
    print(result if result != -1 else -1)

if __name__ == "__main__":
    main()