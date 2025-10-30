import sys

def can_remove_one_char(long_str, short_str):
    n_long = len(long_str)
    n_short = len(short_str)
    if n_long != n_short + 1:
        return False
    i = 0
    j = 0
    skipped = False
    while j < n_short:
        if i < n_long and long_str[i] == short_str[j]:
            i += 1
            j += 1
        else:
            if skipped:
                return False
            skipped = True
            i += 1
    return True

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print("No")
        return
    
    K = int(data[0].strip())
    S = data[1].strip()
    T = data[2].strip()
    
    if S == T:
        print("Yes")
        return
        
    n = len(S)
    m = len(T)
    
    if n == m:
        diff_count = 0
        for i in range(n):
            if S[i] != T[i]:
                diff_count += 1
                if diff_count > 1:
                    break
        print("Yes" if diff_count == 1 else "No")
    elif n == m + 1:
        result = can_remove_one_char(S, T)
        print("Yes" if result else "No")
    elif n + 1 == m:
        result = can_remove_one_char(T, S)
        print("Yes" if result else "No")
    else:
        print("No")

if __name__ == "__main__":
    main()