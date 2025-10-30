def main():
    import sys, math
    # Read all input tokens.
    # In an interactive contest the only input is the line "N L R" and then our queries get answered.
    # For testing purposes the hidden array A may be appended after the initial three numbers.
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    L = int(data[1])
    R = int(data[2])
    
    # For simulation we allow the full hidden sequence to be provided.
    # The hidden sequence A (of length 2^N) consists of integers between 0 and 99.
    # In a real interactive environment this will not be present.
    hidden_A = None
    if len(data) >= 3 + (1 << N):
        hidden_A = list(map(int, data[3:3 + (1 << N)]))
    
    # The interactive query function.
    # We are allowed to ask questions on any dyadic block, that is for non-negative integers i and j satisfying:
    #   2^i * (j+1) <= 2^N.
    # The block queried is:
    #   l = 2^i * j and r = 2^i * (j+1) - 1.
    # The judge returns the remainder modulo 100 of A_l + A_{l+1} + ... + A_r.
    def ask(i, j):
        # In simulation (when hidden_A is given) we compute the answer directly.
        if hidden_A is not None:
            l_block = (1 << i) * j
            r_block = (1 << i) * (j + 1) - 1
            s = sum(hidden_A[l_block:r_block + 1]) % 100
            return s
        else:
            # In an interactive environment, print the query and flush.
            print(f"? {i} {j}", flush=True)
            response = sys.stdin.readline().strip()
            if not response:
                # Terminate if no answer.
                sys.exit(0)
            t = int(response)
            if t == -1:
                # If judge indicates an error, exit immediately.
                sys.exit(0)
            return t

    # Our goal is to determine:
    #   S = (A_L + A_{L+1} + ... + A_R) mod 100
    #
    # It turns out that one can always cover the interval [L, R] with a collection of dyadic (power-of-two sized)
    # intervals such that each one is of the form [2^i * j, 2^i*(j+1)-1]. Moreover, there is a unique minimal decomposition.
    # One standard way is to “greedily” cover [L,R] from left to right.
    #
    # Here is a simple procedure:
    #
    #   current = L
    #   while current <= R:
    #       Let p be the maximum power-of-two such that:
    #           (1) p <= (R - current + 1)  (so that the block fits entirely in [L, R])
    #           (2) current is divisible by p (i.e. current % p == 0)
    #       Then the query interval is [current, current + p - 1].
    #       We obtain its sum modulo 100 by asking the query:
    #           i = log2(p)   and   j = current // p.
    #       Then update current = current + p.
    #
    # This algorithm gives the minimal number of queries needed.
    
    total_mod = 0
    current = L
    while current <= R:
        # Start with the smallest power.
        p = 1
        # We want to make p as large as possible without overshooting [current, R] 
        # and while maintaining that the interval is "aligned", i.e. current % (p*2) == 0.
        while (p * 2) <= (R - current + 1) and (current % (p * 2) == 0):
            p *= 2
        # Convert block size p into the exponent i (since p = 2^i).
        exponent = p.bit_length() - 1
        # The block index j is current divided by p.
        block_index = current // p
        # Ask the query on the block [2^i * j, 2^i*(j+1)-1] and get its modulo–100 sum.
        ans = ask(exponent, block_index)
        total_mod = (total_mod + ans) % 100
        current += p

    # With the answers from the minimal set of queries, we have computed the answer S.
    print(f"! {total_mod}", flush=True)

if __name__ == '__main__':
    main()