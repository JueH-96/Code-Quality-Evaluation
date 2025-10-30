import heapq

def solve():
    H, W = map(int, input().split())
    S = [input() for _ in range(H)]
    A, B, C, D = map(int, input().split())
    A -= 1
    B -= 1
    C -= 1
    D -= 1

    dist = [[float('inf')] * W for _ in range(H)]
    dist[A][B] = 0
    pq = [(0, A, B)]

    while pq:
        d, r, c = heapq.heappop(pq)

        if d > dist[r][c]:
            continue

        # Move
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and S[nr][nc] == '.':
                if dist[nr][nc] > d:
                    dist[nr][nc] = d
                    heapq.heappush(pq, (d, nr, nc))

        # Kick
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            kick_cost = d + 1
            
            if dist[r][c] + 1 < dist[r][c]:
                dist[r][c] = dist[r][c] + 1
                heapq.heappush(pq, (dist[r][c], r, c))

            if 0 <= r < H and 0 <= c < W:
                
                temp_s = [list(row) for row in S]
                
                for i in range(1, 3):
                    kr, kc = r + dr * i, c + dc * i
                    if 0 <= kr < H and 0 <= kc < W and temp_s[kr][kc] == '#':
                        temp_s[kr][kc] = '.'
                        
                for i in range(H):
                    S[i] = "".join(temp_s[i])
                
                for dr_move, dc_move in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr_move, nc_move = r + dr_move, c + dc_move
                    if 0 <= nr_move < H and 0 <= nc_move < W and S[nr_move][nc_move] == '.':
                        if dist[nr_move][nc_move] > kick_cost:
                            dist[nr_move][nc_move] = kick_cost
                            heapq.heappush(pq, (kick_cost, nr_move, nc_move))
                
                for i in range(1,3):
                    kr, kc = r + dr * i, c + dc * i
                    if 0 <= kr < H and 0 <= kc < W:
                        if dist[kr][kc] > kick_cost:
                            if S[kr][kc] == '.':
                                dist[kr][kc] = kick_cost
                                heapq.heappush(pq, (kick_cost, kr, kc))
                            
    print(dist[C][D])

solve()