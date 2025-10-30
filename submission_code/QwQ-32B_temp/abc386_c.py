def main():
    import sys
    K = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    if S == T:
        print("Yes")
        return

    len_S = len(S)
    len_T = len(T)

    if len_S == len_T:
        # Check replace: exactly one difference
        diff = 0
        for s, t in zip(S, T):
            if s != t:
                diff += 1
                if diff > 1:
                    break
        if diff == 1:
            print("Yes")
        else:
            print("No")
    elif abs(len_S - len_T) == 1:
        possible = False
        if len_S < len_T:
            # Check if T can be formed by inserting one into S
            i = j = 0
            used = False
            while i < len_S and j < len_T:
                if S[i] == T[j]:
                    i += 1
                    j += 1
                else:
                    if used:
                        break
                    j += 1
                    used = True
            if i == len_S:
                possible = True
        else:
            # Check if S can be formed by inserting one into T (equivalent to delete from S)
            i = j = 0
            used = False
            while i < len_T and j < len_S:
                if T[i] == S[j]:
                    i += 1
                    j += 1
                else:
                    if used:
                        break
                    j += 1
                    used = True
            if i == len_T:
                possible = True
        if possible:
            print("Yes")
        else:
            print("No")
    else:
        print("No")

if __name__ == "__main__":
    main()