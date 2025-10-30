def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    C = list(map(int, data[2:2+N]))
    
    # Precompute expected T for b=0 and b=1
    expected_T0 = [0 if (i+1) % 2 == 1 else 1 for i in range(N)]
    expected_T1 = [1 if (i+1) % 2 == 1 else 0 for i in range(N)]
    
    # Compute C_b for b=0 and b=1
    C0 = 0
    for i in range(N):
        if S[i] != str(expected_T0[i]):
            C0 += C[i]
    
    C1 = 0
    for i in range(N):
        if S[i] != str(expected_T1[i]):
            C1 += C[i]
    
    INF = float('inf')
    min_total_cost = INF
    
    # Iterate over each possible k from 1 to N-1
    for k in range(N-1):
        # For b=0
        T_k = expected_T0[k]
        T_k1 = expected_T0[k+1]
        
        # Option 1: Flip T_k
        new_Tk = 1 - T_k
        valid1 = True
        if k > 0:
            if expected_T0[k-1] == new_Tk:
                valid1 = False
        if new_Tk != T_k1:
            valid1 = False
        if k+2 < N:
            if T_k1 == expected_T0[k+2]:
                valid1 = False
        cost1 = C[k] if valid1 else INF
        
        # Option 2: Flip T_k1
        new_Tk1 = 1 - T_k1
        valid2 = True
        if T_k != new_Tk1:
            valid2 = False
        if k+2 < N:
            if new_Tk1 == expected_T0[k+2]:
                valid2 = False
        cost2 = C[k+1] if valid2 else INF
        
        additional_cost = min(cost1, cost2)
        if additional_cost != INF:
            total_cost = C0 + additional_cost
            if total_cost < min_total_cost:
                min_total_cost = total_cost
        
        # For b=1
        T_k = expected_T1[k]
        T_k1 = expected_T1[k+1]
        
        # Option 1: Flip T_k
        new_Tk = 1 - T_k
        valid1 = True
        if k > 0:
            if expected_T1[k-1] == new_Tk:
                valid1 = False
        if new_Tk != T_k1:
            valid1 = False
        if k+2 < N:
            if T_k1 == expected_T1[k+2]:
                valid1 = False
        cost1 = C[k] if valid1 else INF
        
        # Option 2: Flip T_k1
        new_Tk1 = 1 - T_k1
        valid2 = True
        if T_k != new_Tk1:
            valid2 = False
        if k+2 < N:
            if new_Tk1 == expected_T1[k+2]:
                valid2 = False
        cost2 = C[k+1] if valid2 else INF
        
        additional_cost = min(cost1, cost2)
        if additional_cost != INF:
            total_cost = C1 + additional_cost
            if total_cost < min_total_cost:
                min_total_cost = total_cost
    
    print(min_total_cost)

if __name__ == '__main__':
    main()