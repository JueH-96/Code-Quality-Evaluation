import heapq

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    C = list(map(int, input().split()))

    # Max-priority queue for C_i where A_i=1, B_i=0 (Type 10)
    # We store negative values in Python's min-heap to simulate a max-heap
    pq_10 = []
    # Min-priority queue for C_i where A_i=0, B_i=1 (Type 01)
    pq_01 = []

    # current_s represents the sum of C_k for all k where A_k is currently 1.
    current_s = 0
    for i in range(N):
        if A[i] == 1:
            current_s += C[i]
            # If A_i is 1 and B_i is 0, this bit must be flipped from 1 to 0.
            # It is currently contributing C_i to the sum, and this contribution must be removed.
            # Flipping 1->0 reduces the sum by C_i.
            heapq.heappush(pq_10, -C[i])
        else: # A[i] == 0
            # If A_i is 0 and B_i is 1, this bit must be flipped from 0 to 1.
            # It is currently not contributing to the sum, but must contribute C_i after the flip.
            # Flipping 0->1 increases the sum by C_i.
            if B[i] == 1:
                heapq.heappush(pq_01, C[i])

    total_cost = 0

    # We need to perform operations until A becomes identical to B.
    # The indices where A_i != B_i are precisely those represented in pq_10 and pq_01.
    # Each operation on an index i where A_i != B_i makes A_i equal to B_i.
    # The problem states that flipping A_i where A_i == B_i an even number of times (including zero)
    # is necessary. The sample explanation suggests only flipping mismatched bits is optimal.
    # We follow this greedy strategy: at each step, perform a flip on a mismatched bit.
    # The cost of an operation is the sum of A_k * C_k AFTER the flip.
    # If we flip A_i from 1 to 0, the new sum is S - C_i. The cost is S - C_i.
    # If we flip A_i from 0 to 1, the new sum is S + C_i. The cost is S + C_i.
    # We want to choose the flip that results in a lower cost (lower new S).

    while pq_10 or pq_01:
        # Option 1: Flip a bit from 1 to 0 (from pq_10 candidates).
        # Choose the one that reduces S the most (largest C_i in pq_10).
        # The cost is S - C_i.
        cost_option1 = float('inf')
        if pq_10:
            c_i_10 = -pq_10[0] # Get the largest C_i from pq_10
            cost_option1 = current_s - c_i_10

        # Option 2: Flip a bit from 0 to 1 (from pq_01 candidates).
        # Choose the one that increases S the least (smallest C_j in pq_01).
        # The cost is S + C_j.
        cost_option2 = float('inf')
        if pq_01:
            c_j_01 = pq_01[0] # Get the smallest C_j from pq_01
            cost_option2 = current_s + c_j_01

        # Compare the costs of the two best available operations and pick the cheaper one.
        if cost_option1 <= cost_option2:
            # Perform Option 1 (Flip 1->0 on the Type 10 bit with largest C_i)
            c_i = -heapq.heappop(pq_10) # Get and remove the largest C_i
            new_s = current_s - c_i
            total_cost += new_s
            current_s = new_s
        else:
            # Perform Option 2 (Flip 0->1 on the Type 01 bit with smallest C_j)
            c_j = heapq.heappop(pq_01) # Get and remove the smallest C_j
            new_s = current_s + c_j
            total_cost += new_s
            current_s = new_s

    print(total_cost)

solve()