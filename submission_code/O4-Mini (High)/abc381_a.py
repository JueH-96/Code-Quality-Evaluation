def main():
    import sys
    input = sys.stdin.readline
    N = int(input().strip())
    S = input().strip()
    # Length must be odd
    if N % 2 == 0:
        print("No")
        return
    # Middle position (1-indexed)
    mid = (N + 1) // 2
    # Check the slash in the middle
    if S[mid - 1] != '/':
        print("No")
        return
    # Check all before are '1'
    for i in range(mid - 1):
        if S[i] != '1':
            print("No")
            return
    # Check all after are '2'
    for i in range(mid, N):
        if S[i] != '2':
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()