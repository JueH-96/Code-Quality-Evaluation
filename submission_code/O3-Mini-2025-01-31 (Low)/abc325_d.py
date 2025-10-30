def main():
    import sys
    input = sys.stdin.readline

    # Read number of products
    N = int(input())
    
    # Each product i has printing time window [T_i, D_i].
    # We want to schedule print times t_i where t_i is in [T_i, D_i]
    # such that after a print at time t, next possible print is at t+1.
    # To maximize number, sort by deadline because choosing earliest deadlines first
    # is optimal (similar to an interval scheduling greedy).
    products = []
    for _ in range(N):
        T, D = map(int, input().split())
        products.append((T, D))
    
    # Sort the products by their deadline (D).
    products.sort(key=lambda x: x[1])
    
    current_time = 0  # earliest time when printing is possible
    count = 0
    for T, D in products:
        # The actual printing time we can schedule is the max of available print time and T 
        scheduled_time = max(current_time, T)
        # The scheduled time must be in the product interval: [T, D]
        if scheduled_time <= D:
            count += 1
            # After printing at scheduled_time, printer requires a 1 microsecond charging time.
            current_time = scheduled_time + 1

    sys.stdout.write(str(count))

if __name__ == '__main__':
    main()