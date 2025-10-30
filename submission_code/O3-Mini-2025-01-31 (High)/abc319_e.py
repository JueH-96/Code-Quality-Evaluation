def main():
    import sys
    import numpy as np
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    # Read N, X, Y
    N = int(next(it))
    X = int(next(it))
    Y = int(next(it))
    
    # There are N-1 bus legs. Each bus leg is given by (P, T).
    bus = [None] * (N - 1)
    for i in range(N - 1):
        P = int(next(it))
        T = int(next(it))
        bus[i] = (P, T)
    
    # Read Q queries
    Q = int(next(it))
    queries = [int(next(it)) for _ in range(Q)]
    
    # We will compute the “composite bus function”
    # The bus transformation is as follows:
    #   If your current time is t, then taking a bus with parameters (P, T)
    #   means you depart at the earliest multiple of P (you may wait)
    #   so that the new time is: t + ((P - (t mod P)) mod P) + T.
    # Because P<=8 and all numbers P divide 840, we work modulo L = 840.
    # In our plan, our state will be the residue r = t mod 840 and a cumulative delay D.
    # That is, if you started with time t (with t ≡ r mod 840) and you have computed a 
    # composite delay f[r], then the arrival time (at the current bus stop)
    # is t + f[r] and the residue is (r + f[r]) mod 840.
    #
    # Initially, no delay is accumulated so f[r] = 0 for all r in 0..839.
    mod_val = 840
    # Precompute an array of numbers 0..839 (the possible remainders).
    r_arr = np.arange(mod_val, dtype=np.int64)
    # Our composite delay function over the bus legs: for each r in 0..839,
    # if you start with time t having t mod 840 = r then after taking all buses
    # your time is t + f_arr[r].
    f_arr = np.zeros(mod_val, dtype=np.int64)
    
    # Process each bus leg in order.
    # For a bus leg with (P, T) the update is:
    #   f[r] <-- f[r] + ( (P - ((r + f[r]) mod P)) mod P + T )
    # Note that (P - (t mod P)) mod P is the waiting time (which is 0 if t mod P is 0)
    # and we can use the fact that (-t) mod P gives exactly the required waiting.
    for (P, T) in bus:
        # For each state r, current effective time is (r + f_arr[r]).
        # Compute it in vectorized form.
        cur = r_arr + f_arr
        # waiting time is computed as (-cur) mod P.
        waiting = (-cur) % P
        # Total additional delay for this bus leg.
        add = waiting + T
        f_arr += add

    # Now answer each query.
    # Takahashi first walks from his house to bus stop 1 in X time.
    # So if he leaves at time q then he arrives at bus stop 1 at time (q + X).
    # Let r = (q + X) mod 840. After taking all the buses the time is:
    #    (q + X) + f_arr[r]
    # Finally he walks from bus stop N to Aoki's house in Y time.
    # Thus the answer is: q + X + f_arr[r] + Y.
    answers = []
    for q in queries:
        t0 = q + X
        r = t0 % mod_val
        # f_arr[r] is a numpy.int64, so convert it to int
        ans = t0 + int(f_arr[r]) + Y
        answers.append(str(ans))
    sys.stdout.write("
".join(answers))
    
if __name__ == '__main__':
    main()