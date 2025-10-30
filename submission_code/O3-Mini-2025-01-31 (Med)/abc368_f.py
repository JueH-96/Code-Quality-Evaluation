def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    A = list(map(int, input_data[1:1+N]))
    max_val = max(A)
    
    # We will compute for each number x (1 <= x <= max_val) its Grundy number.
    # The game for an integer x is: you can move to any proper divisor of x (i.e. a divisor d with d < x).
    # Terminal state: x == 1 has no moves so Grundy(1)=0.
    # Then Grundy(x) = mex({Grundy(d) for d in proper divisors of x}).
    
    # Precompute proper divisors using a sieve-like method.
    # For each d from 1 up to max_val//2, d is a proper divisor of any multiple m = k*d with k>=2.
    divisors = [[] for _ in range(max_val+1)]
    for d in range(1, max_val//2 + 1):
        # All multiples m = k*d where k>=2 and m <= max_val
        for m in range(2*d, max_val+1, d):
            divisors[m].append(d)
    
    # Compute Grundy numbers for each number from 1 to max_val.
    grundy = [0] * (max_val+1)
    # Grundy(1) = 0 (terminal position).
    for x in range(2, max_val+1):
        reachable = set()
        for d in divisors[x]:
            reachable.add(grundy[d])
        mex = 0
        while mex in reachable:
            mex += 1
        grundy[x] = mex

    # In combinatorial game theory, the game is equivalent to a nim-sum of all the positions.
    nim_sum = 0
    for x in A:
        nim_sum ^= grundy[x]
    
    # If nim-sum is nonzero, then the first player (Anna) wins; otherwise Bruno wins.
    output = sys.stdout
    if nim_sum != 0:
        output.write("Anna")
    else:
        output.write("Bruno")
        
if __name__ == '__main__':
    main()