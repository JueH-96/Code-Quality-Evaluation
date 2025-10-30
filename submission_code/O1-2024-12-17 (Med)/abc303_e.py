def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    deg = [0] * (N + 1)
    
    # Read edges and compute degrees
    for _ in range(N - 1):
        u, v = map(int, input().split())
        deg[u] += 1
        deg[v] += 1
    
    # Splus = set of nodes whose degree >= 3   (they must be star centers)
    Splus = [i for i in range(1, N + 1) if deg[i] >= 3]
    
    # X = number of such "big" centers
    X = len(Splus)
    # S = sum of degrees of those centers
    S = sum(deg[i] for i in Splus)
    
    # The formula for how many level-2 centers we must have:
    #   3 * (#S2) = N - X - S
    val = N - X - S
    # By the problem statement, this is guaranteed valid (nonnegative and divisible by 3).
    S2 = val // 3  # number of centers of degree 2
    
    # The levels of the stars are exactly:
    #  - deg[i] for all i in Splus (which is >=3)
    #  - plus S2 copies of 2
    star_levels = [deg[i] for i in Splus] + [2] * S2
    star_levels.sort()
    
    # Output the result
    print(" ".join(map(str, star_levels)))


# Don't forget to call main()
if __name__ == "__main__":
    main()