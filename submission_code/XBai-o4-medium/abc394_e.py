import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    graph = []
    for _ in range(N):
        line = sys.stdin.readline().strip()
        graph.append(list(line))
    
    INF = float('inf')
    # For each source i, compute the answer for all j
    result = [[ -1 for _ in range(N)] for _ in range(N)]
    
    for source in range(N):
        # Initialize distance for this source
        dist = [[[INF for _ in range(27)] for _ in range(27)] for _ in range(N)]
        q = deque()
        # Initial state: (source, 26, 26) with distance 0
        dist[source][26][26] = 0
        q.append((source, 26, 26))
        
        while q:
            u, l_code, r_code = q.popleft()
            current_dist = dist[u][l_code][r_code]
            
            # Iterate all possible edges from u to v
            for v in range(N):
                c = graph[u][v]
                if c == '-':
                    continue
                c_code = ord(c) - ord('a')
                # Process transitions
                new_states = []
                if l_code == 26 and r_code == 26:
                    # Case 1
                    new_states.append((v, c_code, c_code))
                    new_states.append((v, 26, 26))
                elif l_code == 26 and r_code != 26:
                    # Case 2
                    if c_code == r_code:
                        new_states.append((v, 26, 26))
                    else:
                        new_states.append((v, c_code, r_code))
                elif r_code == 26 and l_code != 26:
                    # Case 3
                    if c_code == l_code:
                        new_states.append((v, 26, 26))
                    else:
                        new_states.append((v, l_code, c_code))
                else:
                    # Case 4
                    if c_code == l_code:
                        new_states.append((v, 26, r_code))
                    if c_code == r_code:
                        new_states.append((v, l_code, 26))
                    if c_code != l_code and c_code != r_code:
                        new_states.append((v, l_code, c_code))
                        new_states.append((v, c_code, r_code))
                # Process new_states
                for new_v, new_l, new_r in new_states:
                    if dist[new_v][new_l][new_r] > current_dist + 1:
                        dist[new_v][new_l][new_r] = current_dist + 1
                        q.append((new_v, new_l, new_r))
        # After BFS, collect results for this source
        for j in range(N):
            min_dist = dist[j][26][26]
            if min_dist != INF:
                result[source][j] = int(min_dist)
            else:
                result[source][j] = -1
    
    # Output the result
    for row in result:
        print(' '.join(map(str, row)))

if __name__ == '__main__':
    main()