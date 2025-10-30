N, W = map(int, input().split())
blocks = {}
for i in range(N):
    X, Y = map(int, input().split())
    blocks[i+1] = (X, Y)

Q = int(input())
queries = []
for _ in range(Q):
    T, A = map(int, input().split())
    queries.append((T, A))

def simulate_one_step():
    global blocks
    
    if not blocks:
        return False
    
    old_blocks = blocks.copy()
    
    # Step 1: Remove bottom row if full
    min_y = min(y for x, y in blocks.values())
    bottom_positions = {x for x, y in blocks.values() if y == min_y}
    if len(bottom_positions) == W and set(bottom_positions) == set(range(1, W+1)):
        blocks = {block_id: (x, y) for block_id, (x, y) in blocks.items() if y != min_y}
    
    if not blocks:
        return old_blocks != blocks
    
    # Step 2: Move blocks down
    min_y = min(y for x, y in blocks.values())
    blocks_sorted = sorted(blocks.items(), key=lambda item: item[1][1])
    
    for block_id, (x, y) in blocks_sorted:
        if y == min_y:
            continue
        
        occupied_below = any((x, y-1) == pos for pos in blocks.values())
        if not occupied_below:
            blocks[block_id] = (x, y-1)
    
    return old_blocks != blocks

queries_with_index = [(T, A, i) for i, (T, A) in enumerate(queries)]
queries_with_index.sort()

results = [None] * Q
current_time = 0

for T, A, query_index in queries_with_index:
    while current_time < T:
        changed = simulate_one_step()
        current_time += 1
        if not changed:
            break
    
    results[query_index] = "Yes" if A in blocks else "No"

for result in results:
    print(result)