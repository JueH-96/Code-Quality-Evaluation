def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = list(map(int, data[1:N+1]))
    
    if N == 4:
        # For N=4, the minimal sum is achieved by making the middle two elements symmetric around the midpoint of the first and last
        a, b, c, d = X
        M = (a + d) / 2
        new_b = M + (M - b)
        new_c = M - (c - M)
        sum_coords = a + new_b + new_c + d
        print(int(sum_coords))
    else:
        # For N >= 5, the minimal sum is the sum of the original coordinates
        # Because the operation can only be applied to groups of 4, and it's not possible to reduce the sum further
        print(sum(X))

if __name__ == "__main__":
    main()