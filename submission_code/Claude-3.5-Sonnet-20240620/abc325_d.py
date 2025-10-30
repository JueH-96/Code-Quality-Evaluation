# YOUR CODE HERE
import sys
from heapq import heappush, heappop

def max_printable_products():
    N = int(sys.stdin.readline())
    events = []
    
    for i in range(N):
        T, D = map(int, sys.stdin.readline().split())
        heappush(events, (T, 0, i))  # 0 for start event
        heappush(events, (T + D, 1, i))  # 1 for end event
    
    available = []
    printed = set()
    last_print_time = -float('inf')
    count = 0
    
    while events:
        time, event_type, product = heappop(events)
        
        if event_type == 0:  # Start event
            heappush(available, product)
        else:  # End event
            if product in available:
                available.remove(product)
        
        while available and available[0] in printed:
            heappop(available)
        
        if available and time >= last_print_time + 1:
            product_to_print = heappop(available)
            printed.add(product_to_print)
            last_print_time = time
            count += 1
    
    print(count)

max_printable_products()