def main():
    import sys
    input = sys.stdin.readline

    # Read input
    N = int(input().strip())
    S = input().strip()

    # Check if the length is odd.
    if N % 2 == 0:
        print("No")
        return

    # Calculate the position of the slash (1-indexed).
    mid = (N + 1) // 2  # mid-th character must be '/'
    
    # Check that the mid-th character (0-indexed mid-1) is '/'
    if S[mid - 1] != '/':
        print("No")
        return

    # Check that all characters before the slash are '1'
    for c in S[:mid - 1]:
        if c != '1':
            print("No")
            return

    # Check that all characters after the slash are '2'
    for c in S[mid:]:
        if c != '2':
            print("No")
            return

    print("Yes")

if __name__ == '__main__':
    main()