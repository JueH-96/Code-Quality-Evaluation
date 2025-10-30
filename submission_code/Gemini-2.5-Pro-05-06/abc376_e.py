import sys
import heapq

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A_list = list(map(int, sys.stdin.readline().split()))
    B_list = list(map(int, sys.stdin.readline().split()))

    pairs = []
    for i in range(N):
        pairs.append((A_list[i], B_list[i]))
    
    pairs.sort(key=lambda x: x[0])

    min_total_product = float('inf')

    # Using a min-heap to simulate a max-heap for B_values.
    # We store negative B_values. So pq_neg_B[0] is the negative of the largest B_value.
    pq_neg_B = [] 
    sum_B_in_pq = 0 # Sum of actual (positive) B_values currently in the conceptual max-priority queue

    for j in range(N):
        A_cur, B_cur = pairs[j]

        if K == 1:
            min_total_product = min(min_total_product, A_cur * B_cur)
        else: # K > 1, which implies K-1 >= 1.
            # Phase 1: Calculate product if pq_neg_B is ready
            # pq_neg_B must contain K-1 elements, representing smallest B_values from pairs[0...j-1].
            if len(pq_neg_B) == K - 1:
                # Since K-1 >= 1, pq_neg_B is not empty here.
                current_sum_selected_B = B_cur + sum_B_in_pq
                current_product = A_cur * current_sum_selected_B
                min_total_product = min(min_total_product, current_product)

            # Phase 2: Offer B_cur (from pairs[j]) to the PQ.
            # This B_cur becomes a candidate for the K-1 smallest B_values for future iterations.
            if len(pq_neg_B) < K - 1:
                heapq.heappush(pq_neg_B, -B_cur) # Store negative B_cur
                sum_B_in_pq += B_cur
            else: # len(pq_neg_B) == K - 1. PQ is full.
                  # Since K-1 >= 1, pq_neg_B is not empty, so pq_neg_B[0] is safe to access.
                  # Check if B_cur is smaller than the largest B_value currently in PQ.
                  # The largest B_value in PQ corresponds to -pq_neg_B[0].
                if B_cur < -pq_neg_B[0]: 
                    # B_cur is smaller, so it should replace the current largest B in PQ.
                    removed_B_val_neg = heapq.heappop(pq_neg_B)
                    sum_B_in_pq -= (-removed_B_val_neg) # Subtract the actual (positive) B value
                    
                    heapq.heappush(pq_neg_B, -B_cur)
                    sum_B_in_pq += B_cur
    
    sys.stdout.write(str(min_total_product) + "
")

num_test_cases = int(sys.stdin.readline())
for _ in range(num_test_cases):
    solve()