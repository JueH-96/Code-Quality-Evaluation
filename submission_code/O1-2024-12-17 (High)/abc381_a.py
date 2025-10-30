def main():
    N = int(input().strip())
    S = input().strip()

    # Condition 1: N must be odd
    if N % 2 == 0:
        print("No")
        return

    # Middle index (0-based)
    mid = N // 2

    # Condition 2: The middle character must be '/'
    if S[mid] != '/':
        print("No")
        return

    # Condition 3: All characters before the middle should be '1'
    if any(ch != '1' for ch in S[:mid]):
        print("No")
        return

    # Condition 4: All characters after the middle should be '2'
    if any(ch != '2' for ch in S[mid+1:]):
        print("No")
        return

    # If all checks pass
    print("Yes")

# Do not remove the following line
main()