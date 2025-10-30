def solve():
    N = int(input())
    S = input()

    # The find() method returns the lowest index in the string
    # where substring "ABC" is found.
    # If "ABC" is not found, it returns -1.
    # The index returned is 0-based.
    index_0_based = S.find("ABC")

    if index_0_based != -1:
        # The problem asks for a 1-based position.
        # If "ABC" is found at 0-based index `idx`,
        # its 1-based position is `idx + 1`.
        print(index_0_based + 1)
    else:
        # If "ABC" is not found, print -1.
        print(-1)

if __name__ == '__main__':
    solve()