import heapq
import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    if N <= 1:
        print(0)
        return

    # prev[i] stores the original index of the element to the left of A[i]
    # next[i] stores the original index of the element to the right of A[i]
    # -1 indicates no element to the left, N indicates no element to the right.
    prev = [-1] * N
    next_node = [-1] * N # Renamed to next_node to avoid conflict with built-in next()
    for i in range(N):
        if i > 0:
            prev[i] = i - 1
        if i < N - 1:
            next_node[i] = i + 1

    # is_removed[i] tracks if A[i] has been removed from the sequence
    is_removed = [False] * N

    # Max-heap to store (absolute_difference, left_original_idx, right_original_idx)
    # Stored as (-abs_diff, left_idx, right_idx) to simulate max-heap with heapq (min-heap)
    diff_pq = []

    # Initialize priority queue with all initial adjacent pairs
    for i in range(N - 1):
        diff = abs(A[i] - A[i+1])
        heapq.heappush(diff_pq, (-diff, i, i+1))

    total_score = 0
    
    # We perform floor(N/2) operations
    num_operations = N // 2

    for _ in range(num_operations):
        current_diff = 0
        left_idx = -1
        right_idx = -1

        # Find a valid pair (not already removed) with the maximum difference
        while diff_pq:
            neg_diff, l_idx, r_idx = heapq.heappop(diff_pq)
            
            # Check if both elements in the pair are still in the sequence
            if not is_removed[l_idx] and not is_removed[r_idx]:
                current_diff = -neg_diff
                left_idx = l_idx
                right_idx = r_idx
                break
            # If one or both are removed, this pair is invalid, discard and try next
        
        # If no valid pair found (this theoretically shouldn't happen if N >= 2 and loop count is correct)
        if left_idx == -1: 
            break

        total_score += current_diff

        # Mark elements as removed
        is_removed[left_idx] = True
        is_removed[right_idx] = True

        # Update linked list to reflect removal and potential new adjacency
        prev_of_left = prev[left_idx]
        next_of_right = next_node[right_idx]

        # Link neighbors: element prev_of_left now points to next_of_right
        if prev_of_left != -1:
            next_node[prev_of_left] = next_of_right
        
        # Element next_of_right now points back to prev_of_left
        # next_of_right == N means it's conceptually off the end of the array (no element A[N])
        if next_of_right != -1 and next_of_right != N:
            prev[next_of_right] = prev_of_left
        
        # If a new adjacent pair is formed, add its difference to the priority queue
        # This happens if both prev_of_left and next_of_right are valid indices within A
        if prev_of_left != -1 and next_of_right != -1 and next_of_right != N:
            new_diff = abs(A[prev_of_left] - A[next_of_right])
            heapq.heappush(diff_pq, (-new_diff, prev_of_left, next_of_right))

    sys.stdout.write(str(total_score) + "
")

solve()