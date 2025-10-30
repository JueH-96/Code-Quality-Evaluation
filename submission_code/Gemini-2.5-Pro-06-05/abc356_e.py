# YOUR CODE HERE
import sys

def main():
    """
    Main function to read input, solve the problem, and print the output.
    """
    
    # Fast I/O
    # In competitive programming, reading large inputs line by line can be slow.
    # sys.stdin.readline is generally faster than input().
    
    # Read N from the first line of input
    try:
        # This handles potential empty lines at the end of input
        line = sys.stdin.readline()
        if not line:
            return
        N = int(line)
        # Read the sequence A from the second line of input
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handle potential errors during input reading, though not expected
        # under problem constraints.
        return

    # Maximum possible value of an element in A
    MAX_VAL = 10**6
    
    # 1. Count frequencies
    counts = [0] * (MAX_VAL + 1)
    max_A = 0
    for x in A:
        counts[x] += 1
        if x > max_A:
            max_A = x
            
    # 2. Precompute suffix sums: total_counts_from[x] = sum_{v=x..max_A} C(v)
    total_counts_from = [0] * (max_A + 2)
    for i in range(max_A, 0, -1):
        total_counts_from[i] = total_counts_from[i + 1] + counts[i]
        
    # 3. Calculate S1 (pairs with equal values) and S2 (pairs with unequal values)
    s1 = 0
    for i in range(1, max_A + 1):
        if counts[i] > 1:
            s1 += counts[i] * (counts[i] - 1) // 2
            
    s2 = 0
    for d in range(1, max_A + 1):
        if counts[d] == 0:
            continue
            
        # Calculate U_d = sum_{v=d..max_A} (v//d) * C(v)
        # which is equivalent to sum_{k=1..max_A/d} total_counts_from[k*d]
        u_d = 0
        # Iterate through multiples of d: d, 2d, 3d, ...
        for m in range(d, max_A + 1, d):
            u_d += total_counts_from[m]
            
        # Calculate T_d = sum_{v=d+1..max_A} (v//d) * C(v) = U_d - C(d)
        t_d = u_d - counts[d]
        
        # Add contribution to S2
        s2 += counts[d] * t_d
        
    total_sum = s1 + s2
    print(total_sum)

if __name__ == "__main__":
    main()