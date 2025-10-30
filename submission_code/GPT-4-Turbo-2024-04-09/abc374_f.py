def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    X = int(data[2])
    T = list(map(int, data[3:]))
    
    # We will use a greedy approach to minimize dissatisfaction
    # We will process orders in the order of their placement days T_i
    
    dissatisfaction = 0
    current_day = 0
    
    i = 0
    while i < N:
        # Find the next possible shipping day
        next_ship_day = max(current_day + X, T[i])
        
        # Ship as many as K orders starting from this day
        j = i
        while j < N and j < i + K and T[j] <= next_ship_day:
            dissatisfaction += next_ship_day - T[j]
            j += 1
        
        # Update the current day to the day we just shipped
        current_day = next_ship_day
        
        # Move to the next order to be processed
        i = j
    
    print(dissatisfaction)

if __name__ == "__main__":
    main()