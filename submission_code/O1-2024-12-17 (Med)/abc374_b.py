def main():
    import sys
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Compare character by character up to the length of the shorter string
    for i in range(min(len(S), len(T))):
        if S[i] != T[i]:
            print(i + 1)  # 1-based index
            return

    # If all matched so far, the first difference is at the position
    # just beyond the shorter string (if lengths differ).
    if len(S) == len(T):
        print(0)  # No differences found, strings are equal
    else:
        print(min(len(S), len(T)) + 1)

# DO NOT forget to call main()
if __name__ == "__main__":
    main()