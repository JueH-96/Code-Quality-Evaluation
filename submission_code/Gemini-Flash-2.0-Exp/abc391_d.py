def solve():
    N, W = map(int, input().split())
    blocks = []
    for _ in range(N):
        blocks.append(list(map(int, input().split())))
    Q = int(input())
    queries = []
    for _ in range(Q):
        queries.append(list(map(int, input().split())))

    def simulate(time, initial_blocks):
        current_blocks = [block[:] for block in initial_blocks]
        
        for t in range(time):
            # Check if the bottom row is full
            bottom_row_full = True
            for x in range(1, W + 1):
                found = False
                for block in current_blocks:
                    if block[0] == x and block[1] == 1:
                        found = True
                        break
                if not found:
                    bottom_row_full = False
                    break
            
            # Remove bottom row if full
            if bottom_row_full:
                new_blocks = []
                for block in current_blocks:
                    if block[1] > 1:
                        new_blocks.append(block)
                current_blocks = new_blocks
            
            # Move blocks down
            moved = True
            while moved:
                moved = False
                for i in range(len(current_blocks)):
                    
                    can_move = True
                    if current_blocks[i][1] == 1:
                        can_move = False
                    else:
                        for other_block in current_blocks:
                            if other_block != current_blocks[i] and other_block[0] == current_blocks[i][0] and other_block[1] == current_blocks[i][1] - 1:
                                can_move = False
                                break
                    
                    if can_move:
                        current_blocks[i][1] -= 1
                        moved = True
        return current_blocks

    for T, A in queries:
        A -= 1
        final_blocks = simulate(T, blocks)
        
        found = False
        for block in final_blocks:
            if block[0] == blocks[A][0] and block[1] == blocks[A][1]:
                found = True
                break
        
        if found:
            print("Yes")
        else:
            print("No")

solve()