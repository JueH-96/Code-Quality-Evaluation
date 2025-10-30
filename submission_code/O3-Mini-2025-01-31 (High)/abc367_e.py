def main():
    import sys
    # We use sys.stdin.buffer for faster IO.
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Convert the input tokens to integers.
    it = map(int, data)
    N = int(next(it))
    K = int(next(it))
    # Read sequence X and convert to 0-indexed mapping.
    X = [int(next(it)) - 1 for _ in range(N)]
    # Read sequence A.
    A = [int(next(it)) for _ in range(N)]
    
    # If no operations are needed, output the initial sequence.
    if K == 0:
        sys.stdout.write(" ".join(map(str, A)))
        return

    # We'll perform the transformation using doubling (binary exponentiation)
    # The idea is that after one operation, new A[i] = A[ f(i) ]
    # where f(i) = X[i]. After K operations, we have: final_A[i] = A[f^K(i)].
    #
    # We precompute, for each index and each power 2^t (in t from 0 to bit_length(K)-1),
    # the result of applying f 2^t times.
    #
    # Using numpy allows us to vectorize the computation.
    try:
        import numpy as np
        # Create the mapping as a numpy array (0-indexed).
        mapping = np.array(X, dtype=np.int64)
        T = K.bit_length()  # maximum power needed (about 60 for K up to 10^18)
        doubling = [mapping]
        # Precompute doubling table:
        for t in range(1, T):
            # doubling[t] = doubling[t-1] composed with itself.
            doubling.append(doubling[t - 1][doubling[t - 1]])
        # Start with indices 0,1,...,N-1.
        pos = np.arange(N, dtype=np.int64)
        bit = 0
        kk = K
        # For each bit in the binary representation of K, update the positions.
        while kk:
            if kk & 1:
                pos = doubling[bit][pos]
            kk //= 2
            bit += 1
        # Now compute the final sequence.
        A_np = np.array(A, dtype=np.int64)
        final_A = A_np[pos]
        # Print the final sequence.
        sys.stdout.write(" ".join(map(str, final_A.tolist())))
    except ImportError:
        # If numpy is not available, we fall back to plain Python.
        T = K.bit_length()
        doubling = [X]
        for t in range(1, T):
            prev = doubling[t - 1]
            current = [0] * N
            for i in range(N):
                current[i] = prev[prev[i]]
            doubling.append(current)
        res = [0] * N
        for i in range(N):
            pos = i
            ktemp = K
            bit = 0
            while ktemp:
                if ktemp & 1:
                    pos = doubling[bit][pos]
                ktemp //= 2
                bit += 1
            res[i] = A[pos]
        sys.stdout.write(" ".join(map(str, res)))

if __name__ == '__main__':
    main()