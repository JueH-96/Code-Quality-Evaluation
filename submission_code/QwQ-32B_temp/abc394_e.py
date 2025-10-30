import sys

def main():
    N = int(sys.stdin.readline())
    edges = []
    for _ in range(N):
        line = sys.stdin.readline().strip()
        edges.append(list(line))
    
    INF = float('inf')
    dp = [[INF] * (N + 1) for _ in range(N + 1)]
    
    # Initialize diagonal to 0
    for i in range(1, N+1):
        dp[i][i] = 0
    
    # Initialize direct edges
    for u in range(1, N+1):
        for v in range(1, N+1):
            c = edges[u-1][v-1]
            if c != '-':
                dp[u][v] = 1
    
    # Process each character to find paths via pairs
    for c in 'abcdefghijklmnopqrstuvwxyz':
        edges_from = [[] for _ in range(N+1)]
        edges_to = [[] for _ in range(N+1)]
        
        # Build edges_from: for each u, list of x where u->x has char c
        for u in range(1, N+1):
            for v in range(1, N+1):
                if edges[u-1][v-1] == c:
                    edges_from[u].append(v)
        
        # Build edges_to: for each v, list of y where y->v has char c
        for v in range(1, N+1):
            for y in range(1, N+1):
                if edges[y-1][v-1] == c:
                    edges_to[v].append(y)
        
        # Compute min_dp_c for this character
        min_dp_c = [[INF]*(N+1) for _ in range(N+1)]
        for u in range(1, N+1):
            x_list = edges_from[u]
            if not x_list:
                continue
            for v in range(1, N+1):
                y_list = edges_to[v]
                if not y_list:
                    continue
                current_min = INF
                # Iterate through all x in x_list and y in y_list to find min dp[x][y]
                for x in x_list:
                    for y in y_list:
                        if dp[x][y] < current_min:
                            current_min = dp[x][y]
                        if current_min == 0:  # Can't get better than 0
                            break
                    if current_min == 0:
                        break
                if current_min != INF:
                    min_dp_c[u][v] = current_min
        
        # Update dp with the current character's contribution
        for u in range(1, N+1):
            for v in range(1, N+1):
                candidate = 2 + min_dp_c[u][v]
                if candidate < dp[u][v]:
                    dp[u][v] = candidate
    
    # Prepare the output
    for u in range(1, N+1):
        row = []
        for v in range(1, N+1):
            if dp[u][v] == INF:
                row.append(-1)
            else:
                row.append(dp[u][v])
        print(' '.join(map(str, row)))

if __name__ == '__main__':
    main()