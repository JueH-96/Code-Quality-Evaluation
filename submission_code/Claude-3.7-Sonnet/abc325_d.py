def max_products_printed(n, products):
    # Sort products by exit time, and in case of ties, sort by entry time
    products.sort(key=lambda x: (x[0] + x[1], x[0]))
    
    printed_count = 0
    printer_ready_time = 0  # Time when printer is next available
    
    for entry_time, duration in products:
        exit_time = entry_time + duration
        
        # If printer won't be ready before product exits, skip it
        if printer_ready_time > exit_time:
            continue
        
        # Print at the earliest possible time (either when product enters
        # or when printer becomes ready, whichever is later)
        print_time = max(printer_ready_time, entry_time)
        
        # If we can print on the product within its range
        if print_time <= exit_time:
            printed_count += 1
            printer_ready_time = print_time + 1  # Charge for 1 microsecond
    
    return printed_count

n = int(input())
products = []
for _ in range(n):
    t, d = map(int, input().split())
    products.append((t, d))

print(max_products_printed(n, products))