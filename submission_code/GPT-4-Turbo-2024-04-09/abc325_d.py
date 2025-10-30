def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    events = []
    
    index = 1
    for _ in range(N):
        T = int(data[index])
        D = int(data[index + 1])
        events.append((T, 'start'))
        events.append((T + D, 'end'))
        index += 2
    
    # Sort events, with a rule that 'end' comes before 'start' if they are at the same time
    events.sort(key=lambda x: (x[0], x[1] == 'start'))
    
    max_prints = 0
    current_prints = 0
    last_print_time = -1
    
    for time, event_type in events:
        if event_type == 'start':
            # Check if we can print on this product
            if last_print_time < time:
                current_prints += 1
                last_print_time = time + 1  # Update last print time considering the recharge time
        elif event_type == 'end':
            # We only consider ending to potentially reduce the count if it's exactly at the last print time
            if last_print_time == time + 1:
                current_prints -= 1
        
        max_prints = max(max_prints, current_prints)
    
    print(max_prints)

if __name__ == "__main__":
    main()