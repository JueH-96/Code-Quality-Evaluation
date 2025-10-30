def main():
    import sys
    # Read input
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    S = data[1]
    
    # Condition 1: The length of the string must be odd.
    if N % 2 == 0:
        print("No")
        return

    # Compute the middle index (1-indexed) using the formula:
    # mid = (N + 1) // 2, which corresponds to S[mid-1] in 0-index
    mid = (N + 1) // 2

    # Condition 2: 1st through (mid - 1)-th characters
    # must be all '1'
    if mid > 1 and S[:mid-1] != '1' * (mid - 1):
        print("No")
        return

    # Condition 3: Character at position mid must be '/'
    if S[mid - 1] != '/':
        print("No")
        return

    # Condition 4: (mid+1)-th through N-th characters must be all '2'
    if mid < N and S[mid:] != '2' * (N - mid):
        print("No")
        return

    # If all conditions pass
    print("Yes")

if __name__ == '__main__':
    main()