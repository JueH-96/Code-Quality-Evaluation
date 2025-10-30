# YOUR CODE HERE
import sys
import itertools

def main():
    from math import factorial
    from sys import stdin
    import sys
    from itertools import combinations
    input = sys.stdin.read
    data = list(map(int, input().split()))
    grid = data[:9]

    # Define all 8 lines
    lines = [
        [0,1,2], [3,4,5], [6,7,8], # Rows
        [0,3,6], [1,4,7], [2,5,8], # Columns
        [0,4,8], [6,4,2]           # Diagonals
    ]

    # Find lines with exactly two same numbers and one different
    lines_with_constraints = []
    for line in lines:
        nums = [grid[cell] for cell in line]
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        if len(count) == 2:
            # Find which number appears twice
            for num, cnt in count.items():
                if cnt == 2:
                    same_num = num
                    break
            # Identify which cells have the same_num and which have different
            same_cells = []
            diff_cell = -1
            for cell in line:
                if grid[cell] == same_num:
                    same_cells.append(cell)
                else:
                    diff_cell = cell
            if len(same_cells) == 2 and diff_cell != -1:
                lines_with_constraints.append( (same_cells[0], same_cells[1], diff_cell) )

    M = len(lines_with_constraints)
    total_perms = factorial(9)

    # Precompute all subsets
    from collections import defaultdict
    non_disappointed = 0
    for subset in range(1 << M):
        constraints = []
        bits = bin(subset).count('1')
        sign = (-1) ** bits
        for i in range(M):
            if subset & (1 << i):
                c1, c2, c3 = lines_with_constraints[i]
                constraints.append( (c1, c3) )
                constraints.append( (c2, c3) )
        # Count linear extensions with these constraints
        N = 9
        prec = [0]*N
        for u, v in constraints:
            prec[v] |= (1 << u)
        dp_size = 1 << N
        dp = [0] * dp_size
        dp[0] = 1
        for mask in range(dp_size):
            if dp[mask] == 0:
                continue
            # Find all usable elements
            available = []
            for u in range(N):
                if not (mask & (1 << u)) and (prec[u] & ~mask) == 0:
                    available.append(u)
            for u in available:
                dp[mask | (1 << u)] += dp[mask]
        count = dp[dp_size -1]
        non_disappointed += sign * count

    probability = non_disappointed / total_perms
    print(probability)

if __name__ == "__main__":
    main()