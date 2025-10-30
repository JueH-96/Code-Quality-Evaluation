def solve():
    import sys
    K = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    if S == T:
        print("Yes")
        return
    
    len_S = len(S)
    len_T = len(T)
    
    if abs(len_S - len_T) > 1:
        print("No")
        return
    
    if len_S == len_T:
        diff = 0
        for a, b in zip(S, T):
            if a != b:
                diff += 1
                if diff > 1:
                    break
        print("Yes" if diff <= 1 else "No")
    else:
        longer = S if len_S > len_T else T
        shorter = T if len_S > len_T else S
        i = j = 0
        len_short = len(shorter)
        len_long = len(longer)
        while i < len_short and j < len_long:
            if shorter[i] == longer[j]:
                i += 1
                j += 1
            else:
                j += 1
        print("Yes" if i == len_short else "No")

solve()