def main():
    import sys, heapq
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    q = int(next(it))
    
    # A: current values (initially all 0)
    A = [0] * n
    # in_top[i] tells whether index i’s current valid record is in the top group (True) or bottom group (False)
    # (When an update happens, we mark it temporarily as None; lazy removal later will drop stale heap entries.)
    in_top = [False] * n
    
    # We maintain two heaps:
    # top: a min-heap for the k largest elements (by value). Its smallest element (at the heap top) is the threshold.
    # bottom: a max-heap (using negatives) that holds the rest.
    top = []
    bottom = []
    top_size = 0  # count of valid elements in top (we update this manually)
    bottom_size = 0
    sum_top = 0  # sum of values in top
    
    # Initialize: first k indices in top, rest in bottom.
    for i in range(n):
        if i < k:
            in_top[i] = True
            heapq.heappush(top, (0, i))
            top_size += 1
            # sum_top remains 0
        else:
            in_top[i] = False
            heapq.heappush(bottom, (0, i))  # note: -0 == 0, so for zeros it is simple.
            bottom_size += 1

    # Helpers to clean out stale heap entries.
    def clean_top():
        # Remove entries that are no longer valid from the top heap.
        while top:
            val, idx = top[0]
            # An entry is valid if in_top[idx] is True and the stored value equals current A[idx].
            if in_top[idx] is not True or A[idx] != val:
                heapq.heappop(top)
            else:
                break

    def clean_bottom():
        # For bottom heap, we stored (-val, idx). So the actual value is -neg_val.
        while bottom:
            neg_val, idx = bottom[0]
            if in_top[idx] is not False or A[idx] != -neg_val:
                heapq.heappop(bottom)
            else:
                break

    # Helper to push a new record into the top group.
    def push_top(val, idx):
        nonlocal top_size, sum_top
        in_top[idx] = True
        heapq.heappush(top, (val, idx))
        top_size += 1
        sum_top += val

    # Helper to push a new record into the bottom group.
    def push_bottom(val, idx):
        nonlocal bottom_size
        in_top[idx] = False
        heapq.heappush(bottom, (-val, idx))
        bottom_size += 1

    # Process each update.
    out_lines = []
    for _ in range(q):
        # Read update: set A[x] = y (using 0-indexing)
        x = int(next(it)) - 1
        y = int(next(it))
        
        # Remove the old record.
        old_val = A[x]
        if in_top[x] is True:
            top_size -= 1
            sum_top -= old_val
        else:
            bottom_size -= 1
        # Mark it as stale.
        in_top[x] = None
        A[x] = y  # update value
        
        # Insert new record.
        # If the top group is not full, insert into top.
        if top_size < k:
            push_top(y, x)
        else:
            # Otherwise, check the smallest valid element in top.
            clean_top()
            # Note top is non-empty because top_size == k > 0.
            min_top, _ = top[0]
            if y >= min_top:
                push_top(y, x)
            else:
                push_bottom(y, x)
        
        # Rebalance: ensure top group has exactly k records.
        while top_size > k:
            clean_top()
            if not top:
                break
            val, idx = heapq.heappop(top)
            in_top[idx] = False
            top_size -= 1
            sum_top -= val
            push_bottom(val, idx)
        while top_size < k and bottom_size > 0:
            clean_bottom()
            if not bottom:
                break
            neg_val, idx = heapq.heappop(bottom)
            actual_val = -neg_val
            in_top[idx] = True
            top_size += 1
            bottom_size -= 1
            sum_top += actual_val
            heapq.heappush(top, (actual_val, idx))
        
        out_lines.append(str(sum_top))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()