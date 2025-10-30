def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Condition 1: N must be odd
    if N % 2 == 0:
        print("No")
        return

    # Calculate the middle index (1-based -> zero-based)
    mid = (N + 1) // 2

    # Condition 2: Check if the first (mid - 1) characters are '1'
    if S[:mid - 1] != "1" * (mid - 1):
        print("No")
        return

    # Condition 3: The middle character must be '/'
    if S[mid - 1] != "/":
        print("No")
        return

    # Condition 4: The remaining (N - mid) characters must be '2'
    if S[mid:] != "2" * (N - mid):
        print("No")
        return

    # If all conditions are satisfied, print Yes
    print("Yes")

# Do not forget to call main!
if __name__ == "__main__":
    main()