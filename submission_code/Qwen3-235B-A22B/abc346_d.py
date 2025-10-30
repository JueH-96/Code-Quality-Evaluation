import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S_str = data[1]
    S = [int(c) for c in S_str]
    C = list(map(int, data[2:2+N]))
    
    # Precompute prefix sums for left part
    pre_even_0 = [0] * (N + 2)
    pre_even_1 = [0] * (N + 2)
    pre_odd_0 = [0] * (N + 2)
    pre_odd_1 = [0] * (N + 2)
    
    for i in range(N):
        pre_even_0[i+1] = pre_even_0[i]
        if i % 2 == 0:
            if S[i] != 0:
                pre_even_0[i+1] += C[i]
    
    for i in range(N):
        pre_even_1[i+1] = pre_even_1[i]
        if i % 2 == 0:
            if S[i] != 1:
                pre_even_1[i+1] += C[i]
    
    for i in range(N):
        pre_odd_0[i+1] = pre_odd_0[i]
        if i % 2 == 1:
            if S[i] != 0:
                pre_odd_0[i+1] += C[i]
    
    for i in range(N):
        pre_odd_1[i+1] = pre_odd_1[i]
        if i % 2 == 1:
            if S[i] != 1:
                pre_odd_1[i+1] += C[i]
    
    # Precompute suffix sums for right part
    even_0 = [0] * (N + 2)
    even_1 = [0] * (N + 2)
    odd_0 = [0] * (N + 2)
    odd_1 = [0] * (N + 2)
    
    for i in range(N-1, -1, -1):
        even_0[i] = even_0[i+1]
        if i % 2 == 0:
            if S[i] != 0:
                even_0[i] += C[i]
    
    for i in range(N-1, -1, -1):
        even_1[i] = even_1[i+1]
        if i % 2 == 0:
            if S[i] != 1:
                even_1[i] += C[i]
    
    for i in range(N-1, -1, -1):
        odd_0[i] = odd_0[i+1]
        if i % 2 == 1:
            if S[i] != 0:
                odd_0[i] += C[i]
    
    for i in range(N-1, -1, -1):
        odd_1[i] = odd_1[i+1]
        if i % 2 == 1:
            if S[i] != 1:
                odd_1[i] += C[i]
    
    min_cost = float('inf')
    
    for k in range(N-1):
        for t in [0, 1]:
            # Compute b
            pk = k % 2
            if pk == 0:
                b = t
            else:
                b = 1 - t
            
            # Compute left cost
            if b == 0:
                sum_even = pre_even_0[k+1]
                sum_odd = pre_odd_1[k+1]
            else:
                sum_even = pre_even_1[k+1]
                sum_odd = pre_odd_0[k+1]
            left_cost = sum_even + sum_odd
            
            # Compute pair cost
            pair_cost = 0
            if S[k] != t:
                pair_cost += C[k]
            if S[k+1] != t:
                pair_cost += C[k+1]
            
            # Compute right cost
            bit_same = 1 - t
            if pk == 0:
                if bit_same == 0:
                    sum_par_same = even_0[k+2]
                else:
                    sum_par_same = even_1[k+2]
            else:
                if bit_same == 0:
                    sum_par_same = odd_0[k+2]
                else:
                    sum_par_same = odd_1[k+2]
            
            bit_diff = t
            pk_diff = 1 - pk
            if pk_diff == 0:
                if bit_diff == 0:
                    sum_par_diff = even_0[k+2]
                else:
                    sum_par_diff = even_1[k+2]
            else:
                if bit_diff == 0:
                    sum_par_diff = odd_0[k+2]
                else:
                    sum_par_diff = odd_1[k+2]
            
            right_cost = sum_par_same + sum_par_diff
            
            total_cost = left_cost + pair_cost + right_cost
            if total_cost < min_cost:
                min_cost = total_cost
    
    print(min_cost)

if __name__ == "__main__":
    main()