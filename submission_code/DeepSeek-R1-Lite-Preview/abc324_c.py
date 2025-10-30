import sys

def can_insert(S, T):
    i = j = 0
    len_S = len(S)
    len_T = len(T)
    while i < len_S and j < len_T:
        if S[i] == T[j]:
            i += 1
            j += 1
        else:
            j += 1
    return i == len_S and j <= len_T

def can_delete(S, T):
    i = j = 0
    len_S = len(S)
    len_T = len(T)
    while i < len_S and j < len_T:
        if S[i] == T[j]:
            i += 1
            j += 1
        else:
            i += 1
    return j == len_T and i <= len_S

def one_char_change(S, T):
    diff = 0
    for sc, tc in zip(S, T):
        if sc != tc:
            diff += 1
            if diff > 1:
                return False
    return diff <= 1

def main():
    import sys
    data = sys.stdin.read().splitlines()
    N = int(data[0])
    T_prime = data[1]
    S_list = data[2:2+N]
    
    len_T = len(T_prime)
    valid_indices = []
    
    for idx, S in enumerate(S_list, start=1):
        len_S = len(S)
        if len_S == len_T:
            if S == T_prime or one_char_change(S, T_prime):
                valid_indices.append(idx)
        elif len_S == len_T - 1:
            if can_insert(S, T_prime):
                valid_indices.append(idx)
        elif len_S == len_T + 1:
            if can_delete(S, T_prime):
                valid_indices.append(idx)
    
    K = len(valid_indices)
    print(K)
    if K > 0:
        print(' '.join(map(str, valid_indices)))

if __name__ == '__main__':
    main()