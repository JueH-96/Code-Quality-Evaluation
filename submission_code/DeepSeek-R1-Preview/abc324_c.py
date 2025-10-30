import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    T_prime = data[idx]
    idx += 1
    m = len(T_prime)
    S_list = data[idx:idx + N]
    res = []

    for i in range(N):
        S_i = S_list[i]
        n = len(S_i)
        if n not in {m - 1, m, m + 1}:
            continue

        if n == m:
            if S_i == T_prime:
                res.append(i + 1)
                continue
            else:
                diff = 0
                for a, b in zip(S_i, T_prime):
                    if a != b:
                        diff += 1
                        if diff > 1:
                            break
                if diff == 1:
                    res.append(i + 1)
        elif n == m - 1:
            i_ptr = j_ptr = 0
            allowed_skips = 1
            valid = True
            while i_ptr < n and j_ptr < m:
                if S_i[i_ptr] == T_prime[j_ptr]:
                    i_ptr += 1
                    j_ptr += 1
                else:
                    j_ptr += 1
                    allowed_skips -= 1
                    if allowed_skips < 0:
                        valid = False
                        break
            if valid and i_ptr == n and allowed_skips >= 0:
                res.append(i + 1)
        elif n == m + 1:
            i_ptr = j_ptr = 0
            allowed_skips = 1
            valid = True
            while i_ptr < n and j_ptr < m:
                if S_i[i_ptr] == T_prime[j_ptr]:
                    i_ptr += 1
                    j_ptr += 1
                else:
                    i_ptr += 1
                    allowed_skips -= 1
                    if allowed_skips < 0:
                        valid = False
                        break
            if valid and j_ptr == m and allowed_skips >= 0:
                res.append(i + 1)

    print(len(res))
    if res:
        print(' '.join(map(str, res)))

if __name__ == "__main__":
    main()