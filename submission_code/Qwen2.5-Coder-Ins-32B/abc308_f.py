# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    P = list(map(int, input[2:2+N]))
    L = list(map(int, input[2+N:2+N+M]))
    D = list(map(int, input[2+N+M:2+N+2*M]))
    
    # Create a list of tuples (L_i, D_i) and sort by L_i in descending order
    coupons = sorted(zip(L, D), reverse=True)
    
    # Min-heap to keep track of the smallest prices
    min_heap = []
    
    # Total cost
    total_cost = 0
    
    # Process each item
    for price in P:
        # Add the current item price to the heap
        heapq.heappush(min_heap, price)
        
        # Check if we can use a coupon for the current smallest price item
        while coupons and min_heap and min_heap[0] >= coupons[0][0]:
            # Use the coupon
            total_cost += min_heap[0] - coupons[0][1]
            heapq.heappop(min_heap)
            coupons.pop(0)
        else:
            # No coupon can be used, buy at regular price
            total_cost += heapq.heappop(min_heap)
    
    # Add remaining items in the heap (no coupons left)
    while min_heap:
        total_cost += heapq.heappop(min_heap)
    
    print(total_cost)

if __name__ == "__main__":
    main()