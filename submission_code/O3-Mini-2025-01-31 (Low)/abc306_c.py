def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    A = list(map(int, input().split()))
    
    # dict to store counts and second occurrence index for each i
    # We can store f[i] for each number i
    # Instead of list of indices, we can simply count occurrences when scanning
    f = [0] * (N + 1)  # f[i] will eventually hold the middle index (1-based index)
    count = [0] * (N + 1)
    
    # iterate over A, with 1-based index j
    for j, val in enumerate(A, start=1):
        count[val] += 1
        if count[val] == 2:
            f[val] = j
    
    # Create list of pairs (i, f[i]) then sort by f
    result = sorted(range(1, N+1), key=lambda i: f[i])
    
    print(" ".join(map(str, result)))

if __name__ == '__main__':
    main()