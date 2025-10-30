import sys

def main():
    data = sys.stdin.read().splitlines()
    first_line = data[0].split()
    N = int(first_line[0])
    T_prime = first_line[1]
    S_list = data[1: N+1]
    len_T = len(T_prime)
    ans = []
    for idx in range(N):
        S = S_list[idx]
        len_S = len(S)
        if len_S == len_T:
            diff = 0
            for a, b in zip(S, T_prime):
                if a != b:
                    diff += 1
                    if diff > 1:
                        break
            if diff <= 1:
                ans.append(idx + 1)
        elif len_S == len_T - 1:
            i = 0
            for c in T_prime:
                if i < len_S and c == S[i]:
                    i += 1
            if i == len_S:
                ans.append(idx + 1)
        elif len_S == len_T + 1:
            i = 0
            for c in S:
                if i < len_T and c == T_prime[i]:
                    i += 1
            if i == len_T:
                ans.append(idx + 1)
    print(len(ans))
    if ans:
        print(' '.join(map(str, ans)))

if __name__ == "__main__":
    main()