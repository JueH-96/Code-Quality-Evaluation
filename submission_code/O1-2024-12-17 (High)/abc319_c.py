def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    # Read the 3x3 grid in row-major order
    c = []
    for _ in range(3):
        c.extend(map(int, sys.stdin.readline().split()))
    
    # All 8 possible "lines" of 3 squares each (by index in row-major)
    lines = [
        (0,1,2), (3,4,5), (6,7,8),  # rows
        (0,3,6), (1,4,7), (2,5,8),  # columns
        (0,4,8), (2,4,6)            # diagonals
    ]
    
    # lineset[i][j] will hold the set of squares k such that
    # squares i and j share a line with k (forming exactly two same
    # numbers and one different).  If c[i] == c[j] != c[k], then
    # k is added to lineset[i][j] and lineset[j][i].
    lineset = [[set() for _ in range(9)] for _ in range(9)]
    for (x, y, z) in lines:
        if c[x] == c[y] and c[x] != c[z]:
            lineset[x][y].add(z)
            lineset[y][x].add(z)
        if c[y] == c[z] and c[y] != c[x]:
            lineset[y][z].add(x)
            lineset[z][y].add(x)
        if c[z] == c[x] and c[z] != c[y]:
            lineset[z][x].add(y)
            lineset[x][z].add(y)
    
    # Convert each set to a list for faster iteration later
    for i in range(9):
        for j in range(9):
            lineset[i][j] = list(lineset[i][j])
    
    # squares_of_val[v] will be the list of all square indices whose value is v
    squares_of_val = [[] for _ in range(10)]  # values go from 1..9
    for i in range(9):
        squares_of_val[c[i]].append(i)
    
    used = [False]*9    # tracks if a square index is already placed
    pos = [-1]*9        # pos[i] = position of square i in the reading order
    factorial_9 = 362880  # 9!
    count_valid = 0     # how many permutations do not cause disappointment
    
    def dfs(index):
        nonlocal count_valid
        # If we've placed all 9 squares successfully, it's a valid permutation
        if index == 9:
            count_valid += 1
            return
        
        # Try each unused square s as the (index)-th in reading order
        for s in range(9):
            if not used[s]:
                # Check if placing s here causes disappointment
                ok = True
                val_s = c[s]
                # For all squares i that share the same value c[s], if i is used
                # and i appears earlier, we must ensure that any third square k
                # forming a "2-same-1-different" line with i and s was placed
                # before s, to avoid the pattern i< s< k with c[i]=c[s]!=c[k].
                for i in squares_of_val[val_s]:
                    if used[i] and pos[i] < index:
                        # Then for the line-based third squares k in lineset[i][s],
                        # we need pos[k] < index (i.e., k must also have appeared
                        # before s).
                        for k in lineset[i][s]:
                            if (not used[k]) or (pos[k] >= index):
                                ok = False
                                break
                        if not ok:
                            break
                
                # If it's safe, recurse
                if ok:
                    used[s] = True
                    pos[s] = index
                    dfs(index + 1)
                    used[s] = False
                    pos[s] = -1
    
    # Run backtracking
    dfs(0)
    
    # Print the probability (count_valid / 9!)
    # The problem guarantees accuracy if the absolute error is <= 1e-8
    print(f"{count_valid / factorial_9:.30f}")

# Don't forget to call main!
if __name__ == "__main__":
    main()