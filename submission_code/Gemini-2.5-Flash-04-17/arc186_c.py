import heapq
import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    boxes = []
    for _ in range(N):
        V, P = map(int, sys.stdin.readline().split())
        boxes.append((V, P))

    # Mr. Box only considers boxes where V > P for potential positive net earnings if used fully.
    # While he might use a box with V <= P if it helps enable profit from other boxes,
    # the strategy boils down to selecting a set of boxes to maximize total accepted balls minus total cost.
    # The total accepted balls is limited by M and the capacities.

    # Based on the sample cases and common patterns in such competitive games,
    # when Mr. Box chooses a set S of boxes with capacities {V_i | i in S},
    # and assigns these capacities to M types, the total number of balls he can accept
    # before Mr. Ball forces the game to end is sum(V_i for i in S) - sum of the capacities
    # of the M-1 boxes with the smallest capacities in S.
    # This assumes |S| >= M. If |S| < M, the game ends immediately when Mr. Ball gives a type
    # for which Mr. Box has no assigned capacity, leading to 0 accepted balls.
    # Mr. Box aims to maximize (Accepted Balls) - (Total Cost of chosen boxes).
    # He should only consider sets S where he can get positive accepted balls, i.e., |S| >= M.
    # To maximize (sum(V_i for i in S) - sum(V_i for M-1 smallest V_i in S)) - sum(P_i for i in S)
    # which simplifies to (sum(V_i for i in S excluding M-1 smallest V_i)) - sum(P_i for i in S).
    # This means he wants to maximize the sum of (V_i - P_i) for the M largest V_i boxes,
    # plus the sum of (-P_i) for the remaining |S|-M boxes.
    # To maximize this, he should select boxes with large V_i and small P_i.
    # Specifically, he should prioritize boxes that contribute positively to the accepted ball count
    # (the M largest V_i) and minimize the cost for any additional boxes.
    # This means he should select the boxes with the largest V values to be among the top M,
    # and for any boxes beyond the M-th, he should select those with the smallest P values.
    # However, the problem simplifies: consider selecting *any* p boxes. The accepted balls will be the sum
    # of the M largest V's among those p boxes, if p >= M. He maximizes this sum minus the cost of those p boxes.
    # To maximize the sum of the M largest V's for a fixed p, he should select the p boxes with the largest V's.
    # So, sort all boxes by V descending. Iterate through prefixes of this sorted list.

    boxes.sort(key=lambda item: item[0], reverse=True)

    max_profit = 0
    current_cost = 0
    
    # Min-heap to keep track of the M largest V values among the boxes considered so far.
    # The sum of elements in this heap will be the sum of the top min(i+1, M) V values.
    top_m_v_heap = []
    sum_top_m_v = 0

    for i in range(N):
        V, P = boxes[i]
        current_cost += P

        # Maintain heap of size M containing the M largest V values seen among the first i+1 boxes.
        if len(top_m_v_heap) < M:
            heapq.heappush(top_m_v_heap, V)
            sum_top_m_v += V
        elif V > top_m_v_heap[0]:
            min_in_heap = heapq.heapreplace(top_m_v_heap, V)
            sum_top_m_v = sum_top_m_v - min_in_heap + V

        # After considering the first i+1 boxes (sorted by V desc).
        # If we have considered at least M boxes (i.e., i+1 >= M),
        # Mr. Box can potentially assign capacities to M types such that
        # the number of accepted balls is the sum of the M largest capacities
        # among these i+1 boxes.
        # The sum of top M capacities among the first i+1 V-sorted boxes
        # is correctly maintained in sum_top_m_v IF i+1 >= M (heap is full).
        # If i+1 < M, the heap size is i+1, sum_top_m_v is the sum of all i+1 V's.
        # The number of accepted balls is the sum of the capacities of the top min(i+1, M) boxes by V.
        # This sum is exactly sum_top_m_v.

        accepted_balls = sum_top_m_v

        # Calculate profit using the first i+1 boxes by V as the set S.
        # This is only possible if i+1 >= M, otherwise he cannot guarantee positive capacity for M types.
        # However, the number of accepted balls formula implicitly handles this; if i+1 < M,
        # sum_top_m_v is the sum of i+1 V's.
        # The accepted balls should be 0 if Mr. Box cannot set up capacity for M types.
        # This happens if he uses fewer than M boxes with V>0 capacity.
        # But the formula for accepted balls implies it only makes sense for >= M boxes.
        # Let's use the condition len(top_m_v_heap) == M to ensure we have at least M boxes contributing.

        if len(top_m_v_heap) == M:
             profit = accepted_balls - current_cost
             max_profit = max(max_profit, profit)

    # The maximum profit might be 0 if all potential profits are negative.
    print(max(0, max_profit))


T = int(sys.stdin.readline())
for _ in range(T):
    solve()