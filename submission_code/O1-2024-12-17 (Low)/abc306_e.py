def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    # Parse first line
    N, K, Q = map(int, input_data[:3])
    
    # Updates parsed as (X_i, Y_i)
    updates = input_data[3:]
    updates = list(zip(updates[0::2], updates[1::2]))
    updates = [(int(x), int(y)) for x,y in updates]

    # Current values in A
    A = [0]*(N+1)  # 1-indexed; A[i] will hold the current value of element i
    
    # We will keep two heaps:
    #  topK: a min-heap containing up to K of the largest values; we store (val, idx)
    #  rest: a max-heap containing the remainder; we store (-val, idx)
    # We also keep a running sum of all values in topK.
    
    topK = []
    rest = []
    
    # For lazy removal, we always compare the heap's top with the current A[idx].
    # We push new tuples and only when they appear at the top do we check if they're still valid.
    
    # Initialize:
    # Put the first K items in topK (all zero) and the remaining N-K in rest (also zero but separate).
    # This will be O(N) without sorting.
    # We'll store (A[i], i) in topK for i=1..K,
    # and (-A[i], i) in rest for i=K+1..N, since all are 0 initially.
    
    # Edge case: K could be equal to N. Then rest is empty. (That is allowed, so handle carefully.)
    
    for i in range(1, K+1):
        heapq.heappush(topK, (0, i))
    for i in range(K+1, N+1):
        heapq.heappush(rest, (0, i))  # but 0 is stored as ( -0, i ) normally if we want a max-heap
        # Actually, to make "rest" a max-heap, do negative:
        # but pushing 0 won't matter if sign is negative or positive, it's the same
        # but let's unify approach: store negative so we can pop the "largest" from rest
    # Correcting the approach for rest to always store negative of A[i]
    # We need to re-build it carefully (the above might break if we do not store negative)
    rest = []
    for i in range(K+1, N+1):
        # value = 0, store as (-value, i)
        heapq.heappush(rest, (0, i))  # Now "largest" is the smallest negative
        # but 0 is 0. It's okay. We'll just keep consistent usage later.

    sumTop = 0  # sum of values in topK (all zeros now)
    
    # Clean-up functions for lazy removal:
    def clean_topK():
        # Pop from topK while the top doesn't match A
        while topK and topK[0][0] != A[topK[0][1]]:
            heapq.heappop(topK)

    def clean_rest():
        # Pop from rest while the top doesn't match A, remembering rest keeps (negVal, idx)
        while rest and rest[0][0] != -A[rest[0][1]]:
            heapq.heappop(rest)

    # Now define a function to balance topK so it always has K elements if possible.
    def rebalance():
        nonlocal sumTop
        # If topK has fewer than K elements (and there is something in rest), move the largest from rest -> topK
        while len(topK) < K and rest:
            clean_rest()
            if not rest:  # might be empty after cleaning
                break
            neg_val, idx = heapq.heappop(rest)  # this is the largest in rest
            real_val = -neg_val
            sumTop += real_val
            heapq.heappush(topK, (real_val, idx))
        
        # If topK has more than K, pop the smallest from topK -> rest
        while len(topK) > K:
            clean_topK()
            if not topK:
                break
            val, idx = heapq.heappop(topK)
            sumTop -= val
            heapq.heappush(rest, (-val, idx))
    
    # After initialization, if K < N, we already put K zeros in topK, (N-K) zeros in rest
    # sumTop = 0, which is correct for now.
    
    # Process updates
    out = []
    for x, y in updates:
        old_val = A[x]
        A[x] = y
        
        # Cases:
        # 1) If x was in topK, we remove old_val from sumTop and push the ( old_val, x ) for lazy removal.
        #    Then we decide where new_val = y should go. If topK is not full or y is bigger than the min topK value,
        #    it belongs in topK, else in rest.
        # 2) If x was not in topK, it was in rest. We remove the old_val from rest by lazy approach
        #    and then decide the new location of y.
        
        # But to know if x is in topK or rest, we physically check the topK min or rest max?
        # Actually, simpler approach: check if old_val is definitely in topK or rest by comparing with topK's smallest?
        # That isn't guaranteed to know membership. So we do lazy approach: we push a "to-be-removed" version to whichever.
        # The correct method is: we see if old_val <= the smallest value in topK => it must be in rest if there's no duplication?
        # But there could be duplicates. A safer method is to compare "did we see that the topK contained (old_val,x) as valid?" 
        #
        # We'll do the following trick:
        #   If old_val is in topK, then adding one more item bigger than old_val to topK wouldn't pop it out.
        #   Actually, we can do a direct "Which set would old_val have been in if the sets were perfectly cleaned and balanced?"
        # We'll do a simpler approach: We'll remove both possibilities by pushing a lazy removal marker in topK and rest
        # (the correct one will eventually pop). And then we'll proceed as if we're adding y from scratch. 
        #
        # We'll do:
        # sumTop - old_val only if old_val was in topK. In a balanced scenario, old_val is in topK if old_val is >= the smallest topK value or topK not yet full. 
        #
        # Implementation: let's do a small "decide membership" function that checks if old_val is definitely in topK.
        
        if K > 0:
            clean_topK()
            # If topK not full, then old_val was in topK. But initially topK is always full if K <= N.
            # So check if old_val >= the smallest in topK (once cleaned).
            if len(topK) < K:
                # old_val is in topK (since topK wasn't full, so everything so far is in topK)
                sumTop -= old_val
            else:
                # topK is full or eventually will be
                if topK:
                    clean_topK()
                    smallest_val_in_topK = topK[0][0] if topK else 0
                    if old_val >= smallest_val_in_topK:
                        sumTop -= old_val
                # if old_val < smallest_val_in_topK, that means it was in rest, so do nothing to sumTop
        else:
            # If K=0 no topK
            pass
        
        # We push lazy removal markers for old_val in both heaps
        heapq.heappush(topK, (old_val, x))     # "this old_val" is invalid if it doesn't match A[x] now
        heapq.heappush(rest, (-old_val, x))
        
        # Now we add the new_val = y into the correct set if it belongs to topK or rest
        if K > 0:
            clean_topK()
            if len(topK) < K:
                # we can just put y into topK
                sumTop += y
                heapq.heappush(topK, (y, x))
            else:
                # Compare with the smallest in topK
                clean_topK()
                smallest_val_in_topK = topK[0][0] if topK else 0
                if y > smallest_val_in_topK:
                    # pop from topK -> move to rest
                    val_s, idx_s = heapq.heappop(topK)
                    sumTop -= val_s
                    heapq.heappush(rest, (-val_s, idx_s))
                    # push new val in topK
                    sumTop += y
                    heapq.heappush(topK, (y, x))
                else:
                    # smaller or equal, so goes to rest
                    heapq.heappush(rest, (-y, x))
        else:
            # if K=0, everything goes to rest, sumTop=0 always
            heapq.heappush(rest, (-y, x))
        
        # Finally, rebalance in case we messed up the size of topK (but we tried to keep it consistent).
        rebalance()
        
        # Clean topK once more to ensure top is valid just in case
        clean_topK()
        
        # Output the sum of topK
        out.append(str(sumTop))
    
    print("
".join(out))


# IMPORTANT: Do not forget to call main()
if __name__ == "__main__":
    main()