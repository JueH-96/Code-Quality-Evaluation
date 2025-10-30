import heapq

def solve():
    h, w, x = map(int, input().split())
    p, q = map(int, input().split())
    p -= 1
    q -= 1
    s = []
    for _ in range(h):
        s.append(list(map(int, input().split())))

    current_strength = s[p][q]
    absorbed = [[False for _ in range(w)] for _ in range(h)]
    absorbed[p][q] = True

    pq = []  # Stores (-strength, row, col)
    in_queue = set()

    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w

    def add_to_pq(strength, r, c):
        if (r, c) not in in_queue:
            heapq.heappush(pq, (-strength, r, c))
            in_queue.add((r, c))

    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = p + dr, q + dc
        if is_valid(nr, nc) and not absorbed[nr][nc] and s[nr][nc] * x < current_strength:
            add_to_pq(s[nr][nc], nr, nc)

    while pq:
        neg_absorb_strength, r, c = heapq.heappop(pq)
        absorb_strength = -neg_absorb_strength
        in_queue.discard((r, c))

        if absorbed[r][c]:
            continue

        if absorb_strength * x < current_strength:
            current_strength += absorb_strength
            absorbed[r][c] = True
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc) and not absorbed[nr][nc] and s[nr][nc] * x < current_strength:
                    add_to_pq(s[nr][nc], nr, nc)

    print(current_strength)

solve()