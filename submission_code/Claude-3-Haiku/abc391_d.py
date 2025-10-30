import sys

def solve():
    N, W = map(int, input().split())
    blocks = []
    for _ in range(N):
        x, y = map(int, input().split())
        blocks.append((x, y))
    Q = int(input())
    queries = []
    for _ in range(Q):
        t, a = map(int, input().split())
        queries.append((t, a))

    for t in range(1, 10**100+1):
        # Remove blocks in the bottom row
        bottom_row = [False] * W
        for x, y in blocks:
            if y == 1:
                bottom_row[x-1] = True
        if all(bottom_row):
            blocks = [block for block in blocks if block[1] > 1]

        # Move blocks downward
        new_blocks = []
        for x, y in blocks:
            if y == 1 or (x, y-1) in blocks:
                new_blocks.append((x, y))
            else:
                new_blocks.append((x, y-1))
        blocks = new_blocks

    # Answer queries
    for t, a in queries:
        if (X_a, Y_a) in blocks:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    solve()