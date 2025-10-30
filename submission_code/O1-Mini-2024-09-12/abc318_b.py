# YOUR CODE HERE
def main():
    import sys
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    rectangles = []
    idx = 1
    for _ in range(N):
        A, B, C, D = map(int, N_and_rest[idx:idx+4])
        rectangles.append((A, B, C, D))
        idx +=4
    grid = [[False]*101 for _ in range(101)]
    for A, B, C, D in rectangles:
        for x in range(A, B):
            for y in range(C, D):
                grid[x][y] = True
    S = sum(grid[x][y] for x in range(101) for y in range(101))
    print(S)

if __name__ == "__main__":
    main()