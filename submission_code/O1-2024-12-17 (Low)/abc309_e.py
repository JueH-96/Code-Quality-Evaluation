def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    
    # Read N (number of people) and M (number of insurances)
    N, M = map(int, input().split())
    
    # p[i] is the parent of person i+2
    # We'll build an adjacency list of children for each parent
    parents = list(map(int, input().split()))
    children = [[] for _ in range(N+1)]
    for i in range(2, N+1):
        p = parents[i-2]
        children[p].append(i)
    
    # cover[i] := the maximum "remaining generations of coverage" at person i
    # Initialize to -1 (meaning no coverage)
    cover = [-1] * (N+1)
    
    # Read and apply each insurance: if person x_i is covered for y_i generations,
    # then cover[x_i] = max(cover[x_i], y_i).
    for _ in range(M):
        x_i, y_i = map(int, input().split())
        if y_i > cover[x_i]:
            cover[x_i] = y_i
    
    # Propagate coverage down the tree: if cover[i] >= 0,
    # then for each child c, cover[c] = max(cover[c], cover[i] - 1).
    # Since p_i < i, we can iterate from 1..N in ascending order.
    for i in range(1, N+1):
        remaining_cover = cover[i]
        if remaining_cover >= 0:
            for c in children[i]:
                if remaining_cover - 1 > cover[c]:
                    cover[c] = remaining_cover - 1
    
    # Count how many people have cover[i] >= 0 (meaning covered)
    answer = sum(1 for i in range(1, N+1) if cover[i] >= 0)
    
    # Print the result
    print(answer)

# Do not forget to call the main() function
if __name__ == "__main__":
    main()