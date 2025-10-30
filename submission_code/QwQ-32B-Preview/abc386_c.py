def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    K = int(data[0])
    S = data[1].strip()
    T = data[2].strip()

    len_s = len(S)
    len_t = len(T)

    if len_s == len_t:
        # Check if they are equal or differ by one character
        diff_count = sum(1 for a, b in zip(S, T) if a != b)
        if diff_count <= 1:
            print("Yes")
        else:
            print("No")
    elif len_s == len_t + 1:
        # Try deleting one character from S to match T
        i = 0
        j = 0
        deletion_used = False
        while i < len_s and j < len_t:
            if S[i] == T[j]:
                i += 1
                j += 1
            else:
                if deletion_used:
                    print("No")
                    return
                i += 1  # Delete S[i]
                deletion_used = True
        # Check remaining part
        if i < len_s:
            if deletion_used:
                print("No")
                return
            else:
                deletion_used = True
        print("Yes" if deletion_used else "No")
    elif len_s == len_t - 1:
        # Try inserting one character into S to match T
        i = 0
        j = 0
        insertion_used = False
        while i < len_s and j < len_t:
            if S[i] == T[j]:
                i += 1
                j += 1
            else:
                if insertion_used:
                    print("No")
                    return
                j += 1  # Insert T[j] into S
                insertion_used = True
        # Check remaining part
        if j < len_t:
            if insertion_used:
                print("No")
                return
            else:
                insertion_used = True
        print("Yes" if insertion_used else "No")
    else:
        print("No")

if __name__ == "__main__":
    main()