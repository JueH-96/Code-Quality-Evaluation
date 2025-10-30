def solve():
    n, k = map(int, input().split())
    s = input()
    blocks_indices = []
    start_index = -1
    for i in range(n):
        if s[i] == '1':
            if start_index == -1:
                is_start_of_block = (i == 0 or s[i-1] == '0')
                if is_start_of_block:
                    start_index = i
        else:
            if start_index != -1:
                is_end_of_block = (i == n or s[i] == '0')
                if is_end_of_block:
                    end_index = i - 1
                    blocks_indices.append((start_index, end_index))
                    start_index = -1
    if start_index != -1:
        blocks_indices.append((start_index, n-1))
        
    block_to_move_start, block_to_move_end = blocks_indices[k-1]
    insert_after_end = blocks_indices[k-2][1]
    
    t = s[:insert_after_end+1] + s[block_to_move_start:block_to_move_end+1] + s[insert_after_end+1:block_to_move_start] + s[block_to_move_end+1:]
    print(t)

if __name__ == '__main__':
    solve()