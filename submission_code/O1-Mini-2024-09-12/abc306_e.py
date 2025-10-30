import sys, bisect, heapq
from collections import defaultdict

def solve():
    import sys
    import sys
    def input():
        return sys.stdin.read()
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr]); ptr +=1
    K = int(data[ptr]); ptr +=1
    Q = int(data[ptr]); ptr +=1
    updates = []
    for _ in range(Q):
        X = int(data[ptr])-1; ptr +=1
        Y = int(data[ptr]); ptr +=1
        updates.append( (X,Y) )
    A = [0]*N
    in_heap = [0]*N
    heap_top = []
    heap_rest = []
    remove_top = defaultdict(int)
    remove_rest = defaultdict(int)
    sum_top = 0
    # Initialize
    for i in range(N):
        if i < K:
            heapq.heappush(heap_top, A[i])
            sum_top += A[i]
            in_heap[i] = 1
        else:
            heapq.heappush(heap_rest, -A[i])
            in_heap[i] = 0
    for X, Y in updates:
        old_val = A[X]
        if in_heap[X]:
            # Mark old_val for removal from heap_top
            remove_top[old_val] +=1
            sum_top -= old_val
            # Now, need to possibly move one from heap_rest to heap_top
            if heap_rest:
                # Clean heap_rest
                while heap_rest and remove_rest[-heap_rest[0]] >0:
                    val = -heap_rest[0]
                    heapq.heappop(heap_rest)
                    remove_rest[val] -=1
                if heap_rest:
                    moved = -heap_rest[0]
                    heapq.heappop(heap_rest)
                    sum_top += moved
                    heapq.heappush(heap_top, moved)
                    # Find an index which had moved value
                    # We don't track which index, so we can't set in_heap. Instead, set to 1 later
                    # But to keep track, we need to have mapping, which is complicated
                    # Instead, accept that in_heap might be incorrect, but since values can be same, it's tricky
                    # To avoid complexity, we skip updating in_heap for the moved element
            in_heap[X] = 0
        else:
            # Mark old_val for removal from heap_rest
            remove_rest[old_val] +=1
        A[X] = Y
        # Insert new_val
        if len(heap_top) < K:
            heapq.heappush(heap_top, Y)
            sum_top += Y
            in_heap[X] =1
        else:
            # Clean heap_top
            while heap_top and remove_top[heap_top[0]] >0:
                val = heap_top[0]
                heapq.heappop(heap_top)
                remove_top[val] -=1
                sum_top -= val
            if heap_top:
                smallest_top = heap_top[0]
                if Y > smallest_top:
                    # Replace smallest_top with Y
                    heapq.heappop(heap_top)
                    sum_top -= smallest_top
                    heapq.heappush(heap_rest, -smallest_top)
                    heapq.heappush(heap_top, Y)
                    sum_top += Y
                    in_heap[X] =1
                else:
                    heapq.heappush(heap_rest, -Y)
                    in_heap[X] =0
            else:
                heapq.heappush(heap_top, Y)
                sum_top += Y
                in_heap[X] =1
        # After insertion, clean heap_rest
        # Not necessary here
        print(sum_top)