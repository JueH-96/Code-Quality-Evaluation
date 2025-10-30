def main():
    import sys

    data = sys.stdin.read().split()
    # First token is N (we don't really need it), second is the string S
    if len(data) < 2:
        print("No")
        return

    S = data[1]
    n = len(S)

    # Check for adjacent 'a' and 'b' in either order
    for i in range(n - 1):
        if (S[i] == 'a' and S[i+1] == 'b') or (S[i] == 'b' and S[i+1] == 'a'):
            print("Yes")
            return

    print("No")

if __name__ == "__main__":
    main()