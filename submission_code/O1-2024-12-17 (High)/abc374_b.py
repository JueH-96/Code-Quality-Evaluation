def main():
    S = input().strip()
    T = input().strip()

    # If the strings are exactly the same, print 0
    if S == T:
        print(0)
        return

    # Compare characters up to the length of the shorter string
    min_len = min(len(S), len(T))
    for i in range(min_len):
        if S[i] != T[i]:
            # The first differing position (1-based index)
            print(i + 1)
            return

    # No difference was found in the overlapping range,
    # so the difference is at position min_len+1
    print(min_len + 1)

# Do not forget to call main() at the end
if __name__ == "__main__":
    main()