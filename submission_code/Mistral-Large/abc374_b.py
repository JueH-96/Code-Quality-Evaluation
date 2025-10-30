import sys

def find_first_difference(S, T):
    len_S = len(S)
    len_T = len(T)
    min_len = min(len_S, len_T)

    for i in range(min_len):
        if S[i] != T[i]:
            return i + 1

    if len_S == len_T:
        return 0
    else:
        return min_len + 1

def main():
    input = sys.stdin.read
    data = input().split()

    S = data[0]
    T = data[1]

    result = find_first_difference(S, T)
    print(result)

if __name__ == "__main__":
    main()