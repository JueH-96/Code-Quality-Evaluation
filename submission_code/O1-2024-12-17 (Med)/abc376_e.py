def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx_data = 1
    
    out = []
    # We'll process each test case one by one
    for _ in range(t):
        N = int(input_data[idx_data]); idx_data += 1
        K = int(input_data[idx_data]); idx_data += 1
        
        A = list(map(int, input_data[idx_data: idx_data + N]))
        idx_data += N
        B = list(map(int, input_data[idx_data: idx_data + N]))
        idx_data += N
        
        # Solve one test case
        # 1) Pair up (A_i, B_i) and sort by A_i ascending
        pairs = sorted((A[i], B[i]) for i in range(N))
        
        # 2) Group by distinct A_i
        blocks = []
        current_A = None
        block_B = []
        
        for (a_val, b_val) in pairs:
            if a_val != current_A:
                # flush old block if exists
                if block_B:
                    blocks.append((current_A, block_B))
                current_A = a_val
                block_B = []
            block_B.append(b_val)
        # flush the final block
        if block_B:
            blocks.append((current_A, block_B))
        
        # 3) We maintain two heaps:
        #    selected (max-heap) of size up to K  -- store as (-b, block_id)
        #    others (min-heap)  -- store as (b, block_id)
        # But we only need block_id to check if a certain block is used.
        # We'll assign block IDs = index of blocks in the "blocks" list.
        # Then we proceed in ascending order of block index.
        
        # Actually, we only need to know if we have at least one from the current block in "selected".
        # We'll keep a count array for how many from each block are in selected.
        
        selected = []  # max-heap (store (-b, blockID))
        others = []    # min-heap (store (b, blockID))
        heapq.heapify(selected)
        heapq.heapify(others)
        
        sum_selected = 0
        count_in_selected = [0]*len(blocks)
        
        best_answer = 10**20  # a large number
        
        # Add blocks one by one
        for block_id, (valA, blist) in enumerate(blocks):
            # block_id is the index in blocks
            # but we stored blocks as (valueA, B_list)
            # so "valA = blocks[block_id][0]", "blist = blocks[block_id][1]"
            # We'll add them to 'others'
            for b_val in blist:
                heapq.heappush(others, (b_val, block_id))
            
            # Rebalance to ensure "selected" is the K smallest so far
            while len(selected) < K and others:
                b_val, bid = heapq.heappop(others)
                sum_selected += b_val
                count_in_selected[bid] += 1
                # push into selected as (-b_val, bid)
                heapq.heappush(selected, (-b_val, bid))
            
            # We might be able to swap between selected and others to get smaller sums
            # as long as there's something in selected that is bigger than something in others
            while selected and others and -selected[0][0] > others[0][0]:
                # pop largest from selected
                b_sel, bid_sel = heapq.heappop(selected)
                b_sel = -b_sel
                sum_selected -= b_sel
                count_in_selected[bid_sel] -= 1
                # pop smallest from others
                b_oth, bid_oth = heapq.heappop(others)
                # push b_sel to others
                heapq.heappush(others, (b_sel, bid_sel))
                # push b_oth to selected
                sum_selected += b_oth
                count_in_selected[bid_oth] += 1
                heapq.heappush(selected, (-b_oth, bid_oth))
            
            # If we still have more than K in selected (unlikely from the above logic, but just for safety)
            while len(selected) > K:
                b_val, bid = heapq.heappop(selected)
                b_val = -b_val
                sum_selected -= b_val
                count_in_selected[bid] -= 1
                heapq.heappush(others, (b_val, bid))
            
            # If selected < K, we can't form a set of size K yet
            if len(selected) < K:
                continue
            
            # Now "selected" is the K smallest from everything encountered so far
            # The block we just added has index 'block_id'. We want to check if 
            # there is at least one from that block in selected.
            
            v = valA
            cost_candidate1 = 10**20
            if count_in_selected[block_id] > 0:
                # valid set with maxA = v
                cost_candidate1 = v * sum_selected
            
            # Or forcibly ensure we pick an element from this block if none is in selected
            # The best element to pick from the block is the smallest b in that block
            # We'll do that only if count_in_selected[block_id] == 0
            # then we forcibly remove the largest from selected (if needed) and add b_min
            cost_candidate2 = 10**20
            if count_in_selected[block_id] == 0:
                b_min = min(blist)
                largest_in_selected = -selected[0][0]  # the largest B in the selected set
                if b_min < largest_in_selected:
                    # compute the sum if we do that forced replacement
                    sum_forced = sum_selected - largest_in_selected + b_min
                    cost_candidate2 = v * sum_forced
            
            best_answer = min(best_answer, cost_candidate1, cost_candidate2)
        
        out.append(str(best_answer))
    
    print("
".join(out))

# Don't forget to call main()
main()