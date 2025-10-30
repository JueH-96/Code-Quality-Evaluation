import sys

def is_one_deletion(a, b):
    if len(a) != len(b) - 1:
        return False
    i = j = 0
    mismatch = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
            mismatch += 1
            if mismatch > 1:
                return False
    return i == len(a) and mismatch <= 1

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    T_prime = data[1]
    S_list = data[2:2 + n]
    len_T = len(T_prime)
    result = []
    for idx in range(n):
        S = S_list[idx]
        if S == T_prime:
            result.append(idx + 1)
            continue
        len_S = len(S)
        delta = len_S - len_T
        valid = False
        if delta == 0:
            diff = 0
            for c1, c2 in zip(S, T_prime):
                if c1 != c2:
                    diff += 1
                    if diff > 1:
                        break
            valid = (diff == 1)
        elif delta == 1:
            valid = is_one_deletion(T_prime, S)
        elif delta == -1:
            valid = is_one_deletion(S, T_prime)
        if valid:
            result.append(idx + 1)
    print(len(result))
    if result:
        print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()