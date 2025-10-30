def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        print("No")
        return
    N = int(data[0])
    if len(data) < 2:
        print("No")
        return
    S = data[1]
    # Length must match N
    if len(S) != N:
        print("No")
        return
    # N must be odd
    if N % 2 == 0:
        print("No")
        return
    # Zero-based index of '/'
    mid = N // 2
    # Check the middle character
    if S[mid] != '/':
        print("No")
        return
    # Check all before are '1'
    if any(ch != '1' for ch in S[:mid]):
        print("No")
        return
    # Check all after are '2'
    if any(ch != '2' for ch in S[mid+1:]):
        print("No")
        return
    print("Yes")

if __name__ == "__main__":
    main()