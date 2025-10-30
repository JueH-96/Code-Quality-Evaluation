import sys

def main() -> None:
    S = sys.stdin.readline().rstrip('
')
    T = sys.stdin.readline().rstrip('
')

    min_len = min(len(S), len(T))

    # Check characters present in both strings
    for i in range(min_len):
        if S[i] != T[i]:
            print(i + 1)          # 1-based index
            return

    # If all common length characters are equal,
    # the first difference is at the next position
    if len(S) != len(T):
        print(min_len + 1)
    else:
        print(0)

if __name__ == "__main__":
    main()