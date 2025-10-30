def main():
    import sys

    # Read all inputs as a list of tokens.
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))
    # Since A is given in non-decreasing order, we can use two-pointers.

    ans = 0
    j = 0
    # For each candidate mochi "top" at index i (with size A[i]),
    # we want to count the number of mochi "bottom" (with index j > i)
    # such that A[j] >= 2 * A[i]
    for i in range(n):
        if j < i + 1:
            j = i + 1
        while j < n and A[j] < 2 * A[i]:
            j += 1
        ans += (n - j)
    
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()