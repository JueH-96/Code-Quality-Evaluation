def main():
    # Read the two strings from standard input.
    S = input().strip()
    T = input().strip()
    
    # If the strings are exactly the same, output 0.
    if S == T:
        print(0)
        return

    # Otherwise, iterate over the indices until the minimum of both lengths.
    # The strings are 1-indexed in the problem statement.
    min_length = min(len(S), len(T))
    for i in range(min_length):
        if S[i] != T[i]:
            print(i + 1)
            return
    
    # If no difference is found in the overlapping region,
    # the first difference is at the next position.
    print(min_length + 1)

# Call the main function.
main()