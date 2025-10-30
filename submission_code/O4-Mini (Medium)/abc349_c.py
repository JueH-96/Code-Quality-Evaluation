import sys

def main():
    data = sys.stdin.read().split()
    if len(data) != 2:
        return
    S, T = data
    # Case 1: T is a 3-letter subsequence of S (after uppercasing)
    j = 0
    for ch in S:
        if j < 3 and ch.upper() == T[j]:
            j += 1
        if j == 3:
            print("Yes")
            return

    # Case 2: T[2] == 'X' and T[0:2] is a subsequence of S (after uppercasing)
    if T[2] == 'X':
        j = 0
        for ch in S:
            if j < 2 and ch.upper() == T[j]:
                j += 1
            if j == 2:
                print("Yes")
                return

    print("No")

if __name__ == "__main__":
    main()