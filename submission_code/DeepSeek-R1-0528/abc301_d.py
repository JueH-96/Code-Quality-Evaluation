def main():
    S = input().strip()
    N_val = int(input().strip())
    L = len(S)
    
    min_str = S.replace('?', '0')
    max_str = S.replace('?', '1')
    min_val = int(min_str, 2) if min_str != '' else 0
    max_val = int(max_str, 2) if max_str != '' else 0

    if N_val >= (1 << L):
        print(max_val)
        return
        
    if min_val > N_val:
        print(-1)
        return
        
    if max_val <= N_val:
        print(max_val)
        return
        
    bin_rep = bin(N_val)[2:]
    if len(bin_rep) < L:
        binN = bin_rep.zfill(L)
    else:
        binN = bin_rep
        
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def dp(i, tight):
        if i == L:
            return 0
        if S[i] == '?':
            choices = [1, 0]
        else:
            choices = [int(S[i])]
        for d in choices:
            if tight:
                if d > int(binN[i]):
                    continue
            new_tight = tight and (d == int(binN[i]))
            res = dp(i+1, new_tight)
            if res == -1:
                continue
            current_val = d << (L-1-i)
            total = current_val + res
            return total
        return -1

    result = dp(0, True)
    print(result)

if __name__ == '__main__':
    main()