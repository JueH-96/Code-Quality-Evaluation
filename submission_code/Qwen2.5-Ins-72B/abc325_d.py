import sys
from heapq import heappush, heappop

# Read input
N = int(input())
products = [tuple(map(int, input().split())) for _ in range(N)]

# Sort products by their entry time
products.sort()

# Initialize variables
time = 0
max_prints = 0
available_times = []
current_products = []

# Process each product
for T, D in products:
    # Remove products that have already left the range
    while current_products and current_products[0] <= T:
        heappop(current_products)
    
    # Add the exit time of the current product to the heap
    heappush(current_products, T + D)
    
    # If the current time is less than the entry time of the current product, update the time
    if time < T:
        time = T
    
    # If the printer is available, print on the current product
    if time <= T:
        heappush(available_times, time)
        max_prints += 1
        time += 1
    else:
        # If the printer is not available, check if we can print on a product that is still in range
        if available_times and available_times[0] < current_products[0]:
            heappop(available_times)
            heappush(available_times, time)
            max_prints += 1
            time += 1

print(max_prints)