import sys

def main():
    N = int(sys.stdin.readline())
    X = list(map(int, sys.stdin.readline().split()))

    # Constraints: 4 <= N <= 2 * 10^5.
    # X_1 is X[0] in 0-indexed list.
    initial_X1 = X[0]
    
    # If N=1 (not possible by constraints), sum is X[0].
    # For N > 1, there are N-1 gaps.
    if N == 1:
        print(X[0])
        return
        
    # Calculate initial gaps.
    # In 0-based indexing for the list `gaps`:
    # gaps[k] corresponds to the (k+1)-th gap in 1-based indexing (G_{k+1}).
    # e.g. gaps[0] = X[1]-X[0] is G_1.
    gaps = [X[i+1] - X[i] for i in range(N - 1)]

    # S_odd_gaps stores G_1, G_3, ... (i.e., gaps[0], gaps[2], ...)
    S_odd_gaps = gaps[0::2]
    # S_even_gaps stores G_2, G_4, ... (i.e., gaps[1], gaps[3], ...)
    S_even_gaps = gaps[1::2]
            
    S_odd_gaps.sort()
    S_even_gaps.sort()
    
    # final_gaps will store the new sequence G'_1, G'_2, ... G'_{N-1}
    # final_gaps[k] will store G'_{k+1} (the (k+1)-th gap in the optimal arrangement).
    final_gaps = [0] * (N - 1) 
    
    ptr_odd = 0
    ptr_even = 0
    # i iterates from 0 to N-2, representing the 0-based index in final_gaps.
    for i in range(N - 1):
        # if i is 0, final_gaps[0] is G'_1 (an odd-indexed original type gap)
        # if i is 1, final_gaps[1] is G'_2 (an even-indexed original type gap)
        if i % 2 == 0: # Current gap G'_{i+1} is G'_1, G'_3, G'_5...
            final_gaps[i] = S_odd_gaps[ptr_odd]
            ptr_odd += 1
        else: # Current gap G'_{i+1} is G'_2, G'_4, G'_6...
            final_gaps[i] = S_even_gaps[ptr_even]
            ptr_even += 1
            
    # The sum of coordinates X'_1 + ... + X'_N can be expressed as:
    # Sum = N * X'_1 + (N-1)*G'_1 + (N-2)*G'_2 + ... + 1*G'_{N-1}
    # Here X'_1 is initial_X1 (which is invariant).
    # final_gaps[i] stores G'_{i+1}.
    # The coefficient for final_gaps[i] (which is G'_{i+1}) is (N-(i+1)).
    # (N-(i+1)) can be written as (N-1-i).
    
    min_sum = N * initial_X1
    
    for i in range(N - 1): # i from 0 to N-2
        # final_gaps[i] is G'_{i+1}
        # Coefficient is (N - (i+1)) = (N - 1 - i)
        min_sum += (N - 1 - i) * final_gaps[i]
        
    print(min_sum)

if __name__ == '__main__':
    main()