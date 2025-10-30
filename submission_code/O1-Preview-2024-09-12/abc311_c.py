# YOUR CODE HERE
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    A = [0] + A  # Adjust for 1-based indexing
    color = [0] * (N + 1)
    parent = [-1] * (N + 1)

    def dfs(u):
        color[u] = 1  # VISITING
        v = A[u]
        if color[v] == 0:
            parent[v] = u
            dfs(v)
        elif color[v] == 1:
            # Found a cycle
            cycle = [v]
            curr = u
            while curr != v:
                cycle.append(curr)
                curr = parent[curr]
            cycle.append(v)  # Close the cycle
            cycle.reverse()
            print(len(cycle))
            print(' '.join(map(str, cycle)))
            sys.exit()
        # else color[v] == 2, do nothing
        color[u] = 2  # VISITED

    for u in range(1, N+1):
        if color[u] == 0:
            dfs(u)

main()