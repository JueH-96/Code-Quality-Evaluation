def max_printable_products(N, products):
    events = []
    
    # Create events for entering and leaving the printer range
    for i in range(N):
        T, D = products[i]
        events.append((T, 'enter', i))
        events.append((T + D, 'leave', i))
    
    # Sort events by time, with 'enter' events before 'leave' events if they are at the same time
    events.sort(key=lambda x: (x[0], x[1] == 'leave'))
    
    printed_count = 0
    current_time = -1
    active_products = set()
    
    for time, event_type, index in events:
        if event_type == 'enter':
            active_products.add(index)
        elif event_type == 'leave':
            active_products.discard(index)
        
        # Try to print if we can
        if current_time < time:
            if active_products:
                # Print on one of the active products
                printed_count += 1
                current_time = time + 1  # Charge time of 1 microsecond
    
    return printed_count

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N = int(data[0])
    products = [tuple(map(int, line.split())) for line in data[1:N+1]]
    
    result = max_printable_products(N, products)
    print(result)

if __name__ == "__main__":
    main()