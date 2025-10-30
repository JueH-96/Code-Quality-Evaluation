import heapq

def solve():
    N, M = map(int, input().split())
    
    profitable_box_profits = [] # Stores W_i = V_i - P_i
    for _ in range(N):
        v, p = map(int, input().split())
        if v > p:
            profitable_box_profits.append(v - p)
    
    # If there are no profitable boxes, Mr. Box cannot make any profit.
    if not profitable_box_profits:
        print(0)
        return

    # Sort W_i values in descending order to process larger profits first.
    profitable_box_profits.sort(reverse=True)
    
    # Min-priority queue to store current sums of profits for the M types.
    # Initialized with M sums, all zero.
    # heapq structure in Python is a min-heap.
    type_profit_sums_pq = [0] * M
    heapq.heapify(type_profit_sums_pq) # Transform list into a heap in-place. O(M)

    # Distribute W_i values (from profitable_box_profits array).
    # Each W_i is assigned to the type that currently has the minimum total profit.
    for profit_w_i in profitable_box_profits:
        min_sum_for_type = heapq.heappop(type_profit_sums_pq)
        min_sum_for_type += profit_w_i
        heapq.heappush(type_profit_sums_pq, min_sum_for_type)
        
    # The game result will be the minimum sum among the M types,
    # as Mr. Ball will choose this type to minimize Mr. Box's profit.
    # This is the smallest element in the priority queue (at index 0 of the heap array).
    # If K (number of profitable boxes) < M, then M-K sums remained 0 throughout,
    # so type_profit_sums_pq[0] will be 0.
    print(type_profit_sums_pq[0])

# Read the number of test cases
T = int(input())
for _ in range(T):
    solve()