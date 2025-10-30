def main():
    import sys
    data = sys.stdin.read().strip().split()
    S, T = data[0], data[1]  # S is lowercase, T is uppercase

    # Check if T is a length-3 subsequence of S
    def is_subsequence(str_full, str_sub):
        i, j = 0, 0
        while i < len(str_full) and j < len(str_sub):
            if str_full[i] == str_sub[j].lower():
                j += 1
            i += 1
        return j == len(str_sub)

    # Method 1: T is a direct subsequence of S
    if is_subsequence(S, T):
        print("Yes")
        return

    # Method 2: T[:2] is a subsequence of S, and T[2] == 'X'
    if T[2] == 'X' and is_subsequence(S, T[:2]):
        print("Yes")
        return

    print("No")

if __name__ == "__main__":
    main()