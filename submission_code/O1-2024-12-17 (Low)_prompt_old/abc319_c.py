def solve():
    import sys
    sys.setrecursionlimit(10**7)

    # Read inputs
    grid = []
    for _ in range(3):
        row = list(map(int, sys.stdin.readline().split()))
        grid.append(row)
    
    # Flatten into a list of 9 values
    # We'll index them row by row: index = 3*r + c
    cells = [grid[r][c] for r in range(3) for c in range(3)]
    
    # Define the 8 possible "lines" of 3 cells each (using 0-based indexing)
    # lines[i] is a list of 3 cell indices that form a row, column, or diagonal
    lines = [
        [0, 1, 2],  # row0
        [3, 4, 5],  # row1
        [6, 7, 8],  # row2
        [0, 3, 6],  # col0
        [1, 4, 7],  # col1
        [2, 5, 8],  # col2
        [0, 4, 8],  # diag top-left to bottom-right
        [2, 4, 6],  # diag top-right to bottom-left
    ]
    
    # Precompute, for each cell i, which lines contain i
    lines_of_cell = [[] for _ in range(9)]
    for li, line in enumerate(lines):
        for x in line:
            lines_of_cell[x].append(li)
    
    # 9! total permutations
    from math import factorial
    total_perm = factorial(9)
    
    # Each line will be tracked with a small state machine:
    #
    #   count: how many of this line's squares have been seen
    #   firstVal, secondVal: values of the first two squares seen
    #   done: once the line has 2 squares seen that differ, 
    #         or we have placed all 3, no disappointment can happen
    #   inDanger: True if the first two squares seen so far are the same
    #
    # The line triggers "disappointment" if inDanger == True and 
    # the third seen square has a different value.
    
    # We'll backtrack over all permutations (with pruning).
    
    visited = [False]*9  # track which cells are placed in the permutation so far
    
    # line_states[li] = [count, firstVal, secondVal, done, inDanger]
    line_states = [[0, None, None, False, False] for _ in range(8)]
    
    valid_perm_count = 0
    
    def backtrack(pos):
        nonlocal valid_perm_count
        
        if pos == 9:
            # All squares placed without disappointment
            valid_perm_count += 1
            return
        
        # Try placing each unvisited cell as the next in the sequence
        for cell_idx in range(9):
            if not visited[cell_idx]:
                # Temporarily update line states
                old_states = []
                val = cells[cell_idx]
                visited[cell_idx] = True
                can_proceed = True
                
                for li in lines_of_cell[cell_idx]:
                    s = line_states[li]
                    # Save old state
                    old_states.append((li, s[0], s[1], s[2], s[3], s[4]))
                    
                    if s[3]:
                        # already done => no effect
                        continue
                    
                    s[0] += 1  # one more square seen
                    cnt = s[0]
                    if cnt == 1:
                        s[1] = val  # firstVal
                    elif cnt == 2:
                        # secondVal
                        s[2] = val
                        if s[1] == s[2]:
                            # in danger if first two are same
                            s[4] = True  # inDanger
                        else:
                            # if first two differ, line is done (won't cause disappointment)
                            s[3] = True
                    else:
                        # cnt == 3, this is the third square
                        if not s[3]:
                            # not done -> means s[4] might be True
                            if s[4] and val != s[1]:
                                # disappointment
                                can_proceed = False
                            # either we disappointed or the line finishes
                            s[3] = True
                        
                if can_proceed:
                    backtrack(pos+1)
                
                # revert changes
                visited[cell_idx] = False
                for (li, oc, ofv, osv, od, oi) in reversed(old_states):
                    s = line_states[li]
                    s[0] = oc
                    s[1] = ofv
                    s[2] = osv
                    s[3] = od
                    s[4] = oi
    
    backtrack(0)
    
    # Probability = valid_perm_count / 9!
    ans = valid_perm_count / total_perm
    print(f"{ans:.15f}")