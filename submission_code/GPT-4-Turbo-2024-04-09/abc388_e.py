def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Two pointers technique
    i, j = 0, 0
    count = 0
    
    while i < N and j < N:
        # Move j to find a mochi that can be placed on top of mochi at i
        while j < N and A[j] <= 2 * A[i]:
            j += 1
        if j < N:
            # We found a valid pair (i, j)
            count += 1
            # Move i to the next mochi after the current base mochi
            i = j
            # Move j to the next mochi to find the next top mochi
            j += 1
    
    print(count)

if __name__ == "__main__":
    main()