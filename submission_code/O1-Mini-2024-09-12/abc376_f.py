# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    Q = int(data[idx]); idx +=1
    instructions = []
    for _ in range(Q):
        H_i = data[idx]; idx +=1
        T_i = int(data[idx]); idx +=1
        instructions.append((H_i, T_i))
    
    L = 1
    R = 2
    total_cost =0
    for inst in instructions:
        H_i, T_i = inst
        if H_i == 'L':
            H = L
            O = R
        else:
            H = R
            O = L
        
        if H == T_i:
            continue
        else:
            if O == T_i:
                # Move O to adjacent clockwise
                new_O = (T_i % N) +1
                total_cost +=1
                if H_i == 'L':
                    R = new_O
                else:
                    L = new_O
                # Compute distance from H to T_i
                if H <= T_i:
                    d_clock = T_i - H
                else:
                    d_clock = N - H + T_i
                if H >= T_i:
                    d_counter = H - T_i
                else:
                    d_counter = H + N - T_i
                distance = min(d_clock, d_counter)
                total_cost += distance
                # Move H to T_i
                if H_i == 'L':
                    L = T_i
                else:
                    R = T_i
            else:
                # Check if O is on clockwise path from H to T_i
                if H < T_i:
                    on_clockwise_path = (H < O <= T_i)
                else:
                    on_clockwise_path = (O > H or O <= T_i)
                if on_clockwise_path:
                    # Take counter-clockwise path
                    if H >= T_i:
                        d = H - T_i
                    else:
                        d = H + N - T_i
                else:
                    # Take clockwise path
                    if T_i >= H:
                        d = T_i - H
                    else:
                        d = N - H + T_i
                total_cost +=d
                # Move H to T_i
                if H_i == 'L':
                    L = T_i
                else:
                    R = T_i
    print(total_cost)

if __name__ == "__main__":
    main()