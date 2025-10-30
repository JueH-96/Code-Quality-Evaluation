import sys

def main():
    N, K = map(int, sys.stdin.readline().split())
    
    # X_input_1based_vals stores X_1, ..., X_N (1-based values from problem statement)
    # These are indices. X_i means the new B_i gets value from A_{X_i}.
    # In 0-indexed terms: B_new[i] = A_old[X_input_1based_vals[i]-1]
    # This means for each element i of the array, its value is replaced by the value
    # from A_old[g[i]], where g[i] = X_input_1based_vals[i]-1.
    # Let this mapping be g(idx) = g_array[idx].
    # After 1 step: A_1[i] = A_0[g(i)]
    # After 2 steps: A_2[i] = A_1[g(i)] = A_0[g(g(i))]
    # After K steps: A_K[i] = A_0[g^K(i)]
    # g^K(i) is g(g(...g(i)...)) (K applications of g to i).
    
    X_input_1based_vals = list(map(int, sys.stdin.readline().split()))
    A_initial_values = list(map(int, sys.stdin.readline().split()))

    # Convert X to 0-indexed map `g_array`.
    # g_array[i] is the 0-indexed version of X_{i+1} from input.
    g_array = [x - 1 for x in X_input_1based_vals]

    # `f_res` will store g^k(i) for current effective k.
    # Initially, for k=0, g^0(i) = i (identity function).
    # This is the 'result' function being built up.
    f_res = list(range(N))

    # `f_pow` stores g^{2^j}(i) for current bit j.
    # Initially, for j=0, it's g^{2^0}(i) = g^1(i) = g_array[i].
    # This is the 'current power' function.
    f_pow = list(g_array) # Make a copy, g_array is g^1

    # Binary exponentiation for function composition.
    # We want to compute f_res = g^K.
    # Standard algorithm:
    # f_res_current = identity
    # f_pow_current = g
    # while K_val > 0:
    #   if K_val is odd: f_res_current = f_pow_current . f_res_current
    #   f_pow_current = f_pow_current . f_pow_current
    #   K_val = K_val // 2
    
    temp_K = K
    while temp_K > 0:
        if temp_K % 2 == 1:
            # Update f_res: f_res_new(i) = f_pow(f_res(i))
            new_f_res = [0] * N
            for i in range(N):
                new_f_res[i] = f_pow[f_res[i]]
            f_res = new_f_res
        
        # Update f_pow: f_pow_new(i) = f_pow(f_pow(i))
        # This prepares f_pow for the next higher power of 2.
        new_f_pow = [0] * N
        for i in range(N):
            new_f_pow[i] = f_pow[f_pow[i]]
        f_pow = new_f_pow
        
        temp_K //= 2

    # After the loop, f_res[i] stores g^K(i).
    # Construct final A.
    # A_K[i] = A_initial_values[ f_res[i] ]
    # A_initial_values is already 0-indexed as it's a Python list read from input.
    final_A_values = [0] * N
    for i in range(N):
        final_A_values[i] = A_initial_values[f_res[i]]

    sys.stdout.write(" ".join(map(str, final_A_values)) + "
")

if __name__ == '__main__':
    main()