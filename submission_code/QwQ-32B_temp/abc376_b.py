def main():
    import sys
    N, Q = map(int, sys.stdin.readline().split())
    current_L = 1
    current_R = 2
    total = 0

    for _ in range(Q):
        H, T = sys.stdin.readline().split()
        T = int(T)
        if H == 'L':
            blocked = current_R
            current = current_L
            target = T
            if current == target:
                steps = 0
            else:
                L_val = current
                T_val = target
                R_val = blocked
                if T_val > L_val:
                    blocked_clock = (R_val > L_val) and (R_val < T_val)
                else:
                    blocked_clock = (R_val > L_val) or (R_val < T_val)
                
                L2 = target
                T2 = current
                R_val2 = blocked
                if T2 > L2:
                    blocked_counter = (R_val2 > L2) and (R_val2 < T2)
                else:
                    blocked_counter = (R_val2 > L2) or (R_val2 < T2)
                
                d_clockwise = (target - current) % N
                d_counter = N - d_clockwise
                possible = []
                if not blocked_clock:
                    possible.append(d_clockwise)
                if not blocked_counter:
                    possible.append(d_counter)
                steps = min(possible)
            total += steps
            current_L = target
        else:  # H is 'R'
            blocked = current_L
            current = current_R
            target = T
            if current == target:
                steps = 0
            else:
                L_val = current
                T_val = target
                R_val = blocked
                if T_val > L_val:
                    blocked_clock = (R_val > L_val) and (R_val < T_val)
                else:
                    blocked_clock = (R_val > L_val) or (R_val < T_val)
                
                L2 = target
                T2 = current
                R_val2 = blocked
                if T2 > L2:
                    blocked_counter = (R_val2 > L2) and (R_val2 < T2)
                else:
                    blocked_counter = (R_val2 > L2) or (R_val2 < T2)
                
                d_clockwise = (target - current) % N
                d_counter = N - d_clockwise
                possible = []
                if not blocked_clock:
                    possible.append(d_clockwise)
                if not blocked_counter:
                    possible.append(d_counter)
                steps = min(possible)
            total += steps
            current_R = target
    print(total)

if __name__ == "__main__":
    main()