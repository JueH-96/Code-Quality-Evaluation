import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    S = data[0].strip()
    try:
        N = int(data[1].strip())
    except:
        return
    
    L = len(S)
    if N == 0:
        L_N = 1
    else:
        L_N = N.bit_length()
    
    if L > L_N:
        prefix = S[:L - L_N]
        for c in prefix:
            if c == '1':
                print(-1)
                return
        s = S[L - L_N:]
        n_str = bin(N)[2:]
    elif L < L_N:
        s_max = S.replace('?', '1')
        num = int(s_max, 2)
        print(num)
        return
    else:
        s = S
        n_str = bin(N)[2:]
    
    memo = {}
    def dfs(i, tight):
        if i == len(s):
            return 0
        key = (i, tight)
        if key in memo:
            return memo[key]
        
        if s[i] == '?':
            avail = [1, 0]
        else:
            avail = [int(s[i])]
        
        best = -1
        for d in avail:
            if tight:
                n_bit = int(n_str[i])
                if d > n_bit:
                    continue
                new_tight = tight and (d == n_bit)
            else:
                new_tight = False
            
            rest_val = dfs(i + 1, new_tight)
            if rest_val == -1:
                continue
            current_val = (d << (len(s) - 1 - i)) + rest_val
            if current_val > best:
                best = current_val
        
        memo[key] = best
        return best
    
    ans = dfs(0, True)
    print(ans if ans != -1 else -1)

if __name__ == '__main__':
    main()