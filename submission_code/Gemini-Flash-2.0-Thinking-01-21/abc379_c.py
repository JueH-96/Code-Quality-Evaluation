def solve():
    n, m = map(int, input().split())
    x_coords = list(map(int, input().split()))
    a_counts = list(map(int, input().split()))
    initial_stones = {}
    total_initial_stones = 0
    pairs = []
    for i in range(m):
        initial_stones[x_coords[i]] = a_counts[i]
        total_initial_stones += a_counts[i]
        pairs.append((x_coords[i], a_counts[i]))
        
    if total_initial_stones != n:
        print("-1")
        return
        
    sorted_pairs = sorted(pairs, key=lambda p: p[0])
    
    prefix_sum = 0
    pair_index = 0
    for i in range(1, n + 1):
        current_prefix_sum = prefix_sum
        while pair_index < m and sorted_pairs[pair_index][0] <= i:
            current_prefix_sum += sorted_pairs[pair_index][1]
            pair_index += 1
        if current_prefix_sum < i:
            print("-1")
            return
        prefix_sum = current_prefix_sum
        
    current_counts = initial_stones.copy()
    operations = 0
    
    for i in range(1, n):
        stones_in_cell_i = current_counts.get(i, 0)
        move_stones = max(0, stones_in_cell_i - 1)
        operations += move_stones
        current_counts[i] = stones_in_cell_i - move_stones
        if current_counts[i] == 0:
            if i in current_counts:
                del current_counts[i]
        current_counts[i+1] = current_counts.get(i+1, 0) + move_stones
        
    all_cells_one_stone = True
    for i in range(1, n + 1):
        if current_counts.get(i, 0) != 1:
            all_cells_one_stone = False
            break
            
    if all_cells_one_stone:
        print(operations)
    else:
        print("-1")

if __name__ == '__main__':
    solve()