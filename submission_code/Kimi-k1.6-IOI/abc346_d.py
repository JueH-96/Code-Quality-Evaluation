def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    S = input[ptr]
    ptr +=1
    C = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    
    # Precompute prefix costs a0 and a1
    a0 = [0]*N
    a1 = [0]*N
    a0[0] = C[0] if S[0] != '0' else 0
    a1[0] = C[0] if S[0] != '1' else 0
    for j in range(1, N):
        expected0 = '0' if j % 2 == 0 else '1'
        cost0 = C[j] if S[j] != expected0 else 0
        a0[j] = a0[j-1] + cost0
        
        expected1 = '1' if j % 2 == 0 else '0'
        cost1 = C[j] if S[j] != expected1 else 0
        a1[j] = a1[j-1] + cost1
    
    # Precompute suffix costs b0 and b1
    b0 = [0]*N
    b1 = [0]*N
    b0[-1] = C[-1] if S[-1] != '0' else 0
    b1[-1] = C[-1] if S[-1] != '1' else 0
    for j in range(N-2, -1, -1):
        cost0 = C[j] if S[j] != '0' else 0
        cost1 = C[j] if S[j] != '1' else 0
        b0[j] = cost0 + b1[j+1]
        b1[j] = cost1 + b0[j+1]
    
    min_total = float('inf')
    for i in range(N-1):
        for X in [0, 1]:
            Y = 1 - X
            Z = 1 - X
            
            # Compute cost_left
            if i == 0:
                cost_left = 0
            else:
                pos_last = i -1
                if (pos_last) %2 ==0:
                    s = Y
                else:
                    s = 1 - Y
                if s ==0:
                    cost_left = a0[pos_last]
                else:
                    cost_left = a1[pos_last]
            
            # Compute cost_pair
            current_i = S[i]
            current_i1 = S[i+1]
            cost_i = C[i] if (current_i != str(X)) else 0
            cost_i1 = C[i+1] if (current_i1 != str(X)) else 0
            cost_pair = cost_i + cost_i1
            
            # Compute cost_right
            start_right = i +2
            if start_right >= N:
                cost_right =0
            else:
                if Z ==0:
                    cost_right = b0[start_right]
                else:
                    cost_right = b1[start_right]
            
            total = cost_left + cost_pair + cost_right
            if total < min_total:
                min_total = total
    print(min_total)

if __name__ == '__main__':
    main()