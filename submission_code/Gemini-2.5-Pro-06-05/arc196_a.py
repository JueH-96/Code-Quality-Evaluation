import sys
import heapq

def solve():
    """
    Solves the problem by greedily removing adjacent pairs with the maximum
    absolute difference. A min-heap and a doubly linked list (simulated with arrays)
    are used for an efficient implementation.
    """
    try:
        # Use fast I/O for performance
        readline = sys.stdin.readline
        
        N_str = readline()
        if not N_str: return
        N = int(N_str)
        
        A = list(map(int, readline().split()))
    except (IOError, ValueError):
        # Gracefully handle potential empty lines or malformed input
        return

    if N <= 1:
        print(0)
        return

    # Simulate a doubly linked list using arrays
    # nxt[i]: index of the element after A[i]
    # prv[i]: index of the element before A[i]
    # Sentinels: N for end of list, -1 for start of list
    nxt = list(range(1, N + 1))
    prv = list(range(-1, N - 1))
    
    # Track which elements are still in the sequence
    active = [True] * N

    # A max-heap to store adjacent pairs, prioritized by their absolute difference.
    # Implemented using a min-heap on negative differences.
    # Each item: (-absolute_difference, index_of_left_element)
    pq = []
    for i in range(N - 1):
        diff = abs(A[i] - A[i+1])
        heapq.heappush(pq, (-diff, i))

    total_score = 0
    # Perform N // 2 removal operations
    num_ops = N // 2

    for _ in range(num_ops):
        # Pop from heap until a valid pair is found.
        # A pair might be invalid if one of its elements was already removed.
        while True:
            if not pq:
                break
            
            neg_diff, u_idx = heapq.heappop(pq)
            
            # Check if u_idx is active
            if not active[u_idx]:
                continue
            
            v_idx = nxt[u_idx]
            # Check if v_idx is a valid, active element in the sequence
            if v_idx == N or not active[v_idx]:
                continue
            
            # Valid pair found
            break
        else:
            # Heap is empty, but more operations were expected.
            # This is a safeguard and shouldn't be reached in a valid test case.
            break

        # Add the score from the removed pair
        total_score += -neg_diff

        # Mark elements as removed
        active[u_idx] = False
        active[v_idx] = False

        # Get neighbors of the pair (u, v)
        p_idx = prv[u_idx]
        n_idx = nxt[v_idx]

        # Update linked list pointers to bypass the removed elements
        if p_idx != -1:
            nxt[p_idx] = n_idx
        if n_idx != N:
            prv[n_idx] = p_idx

        # If a new adjacent pair is formed, add it to the heap
        if p_idx != -1 and n_idx != N:
            new_diff = abs(A[p_idx] - A[n_idx])
            heapq.heappush(pq, (-new_diff, p_idx))

    print(total_score)

solve()