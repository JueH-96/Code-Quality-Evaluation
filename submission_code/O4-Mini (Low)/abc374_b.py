def main():
    S = input().strip()
    T = input().strip()

    # Iterate up to the maximum length of the two strings
    max_len = max(len(S), len(T))
    for i in range(max_len):
        # If one string is exhausted or characters differ, report position (1-based)
        if i >= len(S) or i >= len(T) or S[i] != T[i]:
            print(i + 1)
            return

    # If no differences found, the strings are identical
    print(0)

if __name__ == "__main__":
    main()