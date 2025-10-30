def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    
    l = 1
    r = 2
    total_ops = 0
    
    for _ in range(Q):
        H = data[idx]
        idx += 1
        T = int(data[idx])
        idx += 1
        
        if H == 'L':
            A = l
            C = r
        else:
            A = r
            C = l
        
        B = T
        
        if A == B:
            steps = 0
        else:
            D_clockwise = (B - A) % N
            D_counter = (A - B) % N
            
            # Check clockwise path
            offset_C_clockwise = (C - A) % N
            clockwise_blocked = False
            if offset_C_clockwise >= 1 and offset_C_clockwise <= D_clockwise - 1:
                clockwise_blocked = True
            
            # Check counter-clockwise path
            offset_C_counter = (C - B) % N
            counter_clockwise_blocked = False
            if offset_C_counter >= 1 and offset_C_counter <= D_counter - 1:
                counter_clockwise_blocked = True
            
            possible = []
            if not clockwise_blocked:
                possible.append(D_clockwise)
            if not counter_clockwise_blocked:
                possible.append(D_counter)
            
            steps = min(possible)
        
        total_ops += steps
        
        if H == 'L':
            l = B
        else:
            r = B
    
    print(total_ops)

if __name__ == '__main__':
    main()