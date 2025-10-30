import collections
import sys

def main():
    N = int(sys.stdin.readline())
    C_matrix = [sys.stdin.readline().strip() for _ in range(N)]

    adj = [[] for _ in range(N)]
    rev_adj = [[] for _ in range(N)]

    for r in range(N):
        for c_idx in range(N):
            char = C_matrix[r][c_idx]
            if char != '-':
                adj[r].append((c_idx, char))
                rev_adj[c_idx].append((r, char))

    infinity = float('inf') 
    # dist[i][j] stores the length of the shortest palindromic path from i to j
    dist = [[infinity] * N for _ in range(N)]

    queue = collections.deque()

    # Base cases: paths of length 0
    # For any vertex i, path i -> i with empty label string (palindrome) has length 0.
    for i in range(N):
        # dist[i][i] is currently infinity. infinity > 0 is true.
        if dist[i][i] > 0: # This check ensures we don't process if already found a shorter path (not strictly needed with current init)
            dist[i][i] = 0
            queue.append((i, i))

    # Base cases: paths of length 1
    # For any edge i --char--> j, path i -> j with label "char" (palindrome) has length 1.
    for r_idx in range(N):
        for neighbor_idx, _ in adj[r_idx]: # char_val not needed here
            # If r_idx == neighbor_idx, dist[r_idx][neighbor_idx] is already 0.
            # The condition dist[r_idx][neighbor_idx] > 1 (i.e. 0 > 1 for self-loops) is false, so dist[r_idx][neighbor_idx] remains 0. Correct.
            if dist[r_idx][neighbor_idx] > 1:
                dist[r_idx][neighbor_idx] = 1
                queue.append((r_idx, neighbor_idx))
    
    # BFS
    # States in queue are (curr_x, curr_y) representing endpoints of a path P: curr_x --> curr_y
    # such that label(P) is a palindrome and length(P) = dist[curr_x][curr_y].
    # We try to extend this to a new path P': u --> v.
    # P' is u --char_match--> curr_x --> ... --> curr_y --char_match--> v
    # label(P') = char_match + label(P) + char_match, which is a palindrome.
    # length(P') = dist[curr_x][curr_y] + 2.
    while queue:
        curr_x, curr_y = queue.popleft()
        d_xy = dist[curr_x][curr_y]

        # Iterate over all incoming edges to curr_x: (u, char1) where u --char1--> curr_x
        for u, char1 in rev_adj[curr_x]:
            # Iterate over all outgoing edges from curr_y: (v, char2) where curr_y --char2--> v
            for v, char2 in adj[curr_y]:
                if char1 == char2: # If labels match, we can extend the palindrome
                    if dist[u][v] > d_xy + 2:
                        dist[u][v] = d_xy + 2
                        queue.append((u, v))
    
    # Prepare final answer matrix
    # (already named ans_matrix in thought process, using dist directly here and converting at end)
    
    # Print results
    # Convert infinity to -1 for output
    for r in range(N):
        row_output = []
        for c_idx in range(N):
            if dist[r][c_idx] == infinity:
                row_output.append("-1")
            else:
                row_output.append(str(int(dist[r][c_idx]))) # Ensure integer output
        sys.stdout.write(" ".join(row_output) + "
")

if __name__ == '__main__':
    main()