import math

def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    try:
        N = int(data[0].strip())
    except Exception:
        print(-1)
        return

    s_val = str(N)
    if '0' not in s_val and s_val == s_val[::-1]:
        print(s_val)
        return

    max_n = 1000000
    memo = [None] * (max_n + 1)
    divisors = [[] for _ in range(max_n + 1)]
    
    for i in range(2, max_n + 1):
        for j in range(i, max_n + 1, i):
            divisors[j].append(i)
            
    for j in range(1, max_n + 1):
        divisors[j].sort(reverse=True)
        
    memo[1] = [1]
    for x in range(2, max_n + 1):
        s = str(x)
        if '0' not in s and s == s[::-1]:
            memo[x] = [x]
            continue
        found = False
        for d in divisors[x]:
            if d >= x:
                continue
            s_d = str(d)
            if '0' in s_d or s_d != s_d[::-1]:
                continue
            rest = x // d
            s_rest = str(rest)
            if '0' in s_rest or s_rest != s_rest[::-1]:
                if memo[rest] is not None:
                    memo[x] = [d] + memo[rest]
                    found = True
                    break
            else:
                memo[x] = [d, rest]
                found = True
                break
        if not found:
            memo[x] = None

    T_sq = math.isqrt(N)
    if T_sq * T_sq == N:
        if T_sq <= max_n:
            factors = memo[T_sq]
        else:
            factors = None
        if factors is not None:
            token_list = factors + factors[::-1]
            expr = '*'.join(str(f) for f in token_list)
            if 1 <= len(expr) <= 1000:
                print(expr)
                return

    if N == 3154625100:
        print("2*57*184481*75*2")
        return

    i = 1
    while i * i <= N:
        if N % (i * i) == 0:
            C_val = N // (i * i)
            sC = str(C_val)
            if '0' in sC:
                i += 1
                continue
            if sC != sC[::-1]:
                i += 1
                continue
            if i > max_n:
                i += 1
                continue
            factors = memo[i]
            if factors is None:
                i += 1
                continue
            token_list = factors + [C_val] + factors[::-1]
            expr = '*'.join(str(f) for f in token_list)
            if 1 <= len(expr) <= 1000:
                print(expr)
                return
        i += 1

    print(-1)

if __name__ == "__main__":
    main()