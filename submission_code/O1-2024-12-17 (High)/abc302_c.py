def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    S = input_data[2:]
    
    # Function to check if two strings differ by exactly one character
    def differ_by_one(s1, s2):
        diff_count = 0
        for x, y in zip(s1, s2):
            if x != y:
                diff_count += 1
                if diff_count > 1:
                    return False
        return diff_count == 1
    
    # Build adjacency matrix: adj[i][j] = True if S[i] differs from S[j] by 1 char
    adj = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j and differ_by_one(S[i], S[j]):
                adj[i][j] = True
    
    # Memoization for DFS
    from functools import lru_cache
    
    @lru_cache(None)
    def dfs(node, visited_mask):
        # If we've visited all strings, return True
        if visited_mask == (1 << N) - 1:
            return True
        # Explore all possible next nodes
        for next_node in range(N):
            if not (visited_mask & (1 << next_node)) and adj[node][next_node]:
                if dfs(next_node, visited_mask | (1 << next_node)):
                    return True
        return False
    
    # Try each string as a starting point
    for start_node in range(N):
        if dfs(start_node, 1 << start_node):
            print("Yes")
            return
    
    print("No")


# Do not forget to call main() at the end
main()