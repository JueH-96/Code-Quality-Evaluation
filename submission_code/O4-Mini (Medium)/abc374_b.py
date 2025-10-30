def main():
    import sys
    data = sys.stdin.read().splitlines()
    S = data[0].strip()
    T = data[1].strip()

    n, m = len(S), len(T)
    k = min(n, m)
    # Compare characters up to the length of the shorter string
    for i in range(k):
        if S[i] != T[i]:
            print(i + 1)
            return
    # All characters up to k are the same
    if n == m:
        print(0)
    else:
        print(k + 1)

# Call the main function
main()