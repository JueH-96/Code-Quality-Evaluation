import sys

def is_valid(S, T_prime):
    m = len(T_prime)
    n = len(S)
    if S == T_prime:
        return True
    if n == m - 1:
        i = j = 0
        while i < n and j < m:
            if S[i] == T_prime[j]:
                i += 1
                j += 1
            else:
                j += 1
        return i == n and (j - i) == 1
    elif n == m + 1:
        i = j = 0
        while i < n and j < m:
            if S[i] == T_prime[j]:
                i += 1
                j += 1
            else:
                i += 1
        return j == m and (i - j) == 1
    elif n == m:
        diff = 0
        for a, b in zip(S, T_prime):
            if a != b:
                diff += 1
                if diff > 1:
                    break
        return diff == 1
    else:
        return False

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    T_prime = data[ptr]
    ptr += 1
    valid_indices = []
    for idx in range(1, N + 1):
        S = data[ptr]
        ptr += 1
        if is_valid(S, T_prime):
            valid_indices.append(idx)
    K = len(valid_indices)
    print(K)
    if K > 0:
        print(' '.join(map(str, valid_indices)))

if __name__ == "__main__":
    main()