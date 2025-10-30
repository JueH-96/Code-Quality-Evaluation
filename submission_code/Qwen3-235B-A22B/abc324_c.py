import sys

def is_one_edit_less(long_str, short_str):
    if len(long_str) != len(short_str) + 1:
        return False
    i = j = 0
    len_l = len(long_str)
    len_s = len(short_str)
    while i < len_l and j < len_s:
        if long_str[i] == short_str[j]:
            i += 1
            j += 1
        else:
            # Skip one character in long_str
            i += 1
            return long_str[i:] == short_str[j:]
    return True

def main():
    input = sys.stdin.read().splitlines()
    first_line = input[0].split()
    N = int(first_line[0])
    T_prime = first_line[1]
    S_list = input[1:N+1]
    
    res = []
    len_T = len(T_prime)
    
    for idx in range(N):
        S = S_list[idx]
        len_S = len(S)
        valid = False
        
        if len_S == len_T:
            if S == T_prime:
                valid = True
            else:
                diff = 0
                for a, b in zip(S, T_prime):
                    if a != b:
                        diff += 1
                        if diff > 1:
                            break
                valid = diff <= 1
        elif len_S == len_T - 1:
            valid = is_one_edit_less(T_prime, S)
        elif len_S == len_T + 1:
            valid = is_one_edit_less(S, T_prime)
        else:
            valid = False
        
        if valid:
            res.append(idx + 1)  # Convert to 1-based index
    
    print(len(res))
    if res:
        print(' '.join(map(str, res)))

if __name__ == "__main__":
    main()