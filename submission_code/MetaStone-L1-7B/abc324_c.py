import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    T = input[idx]
    idx += 1
    S_list = input[idx:idx+N]

    count = 0
    len_T = len(T)

    for S in S_list:
        len_S = len(S)
        if len_S == len_T:
            if S == T:
                count += 1
            else:
                diff = 0
                for a, b in zip(S, T):
                    if a != b:
                        diff += 1
                        if diff > 1:
                            break
                if diff == 1:
                    count += 1
        elif len_S == len_T - 1:
            i = j = 0
            mismatches = 0
            while i < len_T and j < len_S:
                if S[j] == T[i]:
                    j += 1
                    i += 1
                else:
                    mismatches += 1
                    if mismatches > 1:
                        break
                    i += 1
            if j == len_S and mismatches <= 1:
                count += 1
        elif len_S == len_T + 1:
            i = j = 0
            mismatches = 0
            while i < len_T and j < len_S:
                if T[i] == S[j]:
                    i += 1
                    j += 1
                else:
                    mismatches += 1
                    if mismatches > 1:
                        break
                    j += 1
            if i == len_T and mismatches <= 1:
                count += 1

    print(count)
    print(' '.join(map(str, range(1, count + 1))))

if __name__ == "__main__":
    main()