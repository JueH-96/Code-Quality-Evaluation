# YOUR CODE HERE
def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]

    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w

    def check(r1, c1, dr, dc):
        r2, c2 = r1 + dr, c1 + dc
        r3, c3 = r2 + dr, c2 + dc
        r4, c4 = r3 + dr, c3 + dc
        r5, c5 = r4 + dr, c4 + dc
        
        if not all(is_valid(r, c) for r, c in [(r1, c1), (r2, c2), (r3, c3), (r4, c4), (r5, c5)]):
            return None

        chars = ""
        chars += grid[r1][c1]
        chars += grid[r2][c2]
        chars += grid[r3][c3]
        chars += grid[r4][c4]
        chars += grid[r5][c5]
        
        if chars == "snuke":
            return [(r1+1, c1+1), (r2+1, c2+1), (r3+1, c3+1), (r4+1, c4+1), (r5+1, c5+1)]
        else:
            return None

    drs = [-1, -1, -1, 0, 0, 1, 1, 1]
    dcs = [-1, 0, 1, -1, 1, -1, 0, 1]
    
    for r in range(h):
        for c in range(w):
            for dr in drs:
                for dc in dcs:
                    result = check(r, c, dr, dc)
                    if result:
                        for r_ans, c_ans in result:
                            print(r_ans, c_ans)
                        return

solve()