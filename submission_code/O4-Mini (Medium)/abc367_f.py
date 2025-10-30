import sys
import threading

def main():
    import sys
    data = sys.stdin
    # read N, Q
    line = data.readline().split()
    if not line:
        return
    N, Q = map(int, line)
    A = list(map(int, data.readline().split()))
    B = list(map(int, data.readline().split()))
    
    # We'll use double hashing with two large primes to avoid collisions.
    mod1 = 10**9 + 7
    mod2 = 10**9 + 9
    
    # Pre‐generate random hash values for each possible element 1..N
    import random
    random.seed(123456)
    h1 = [0] * (N + 1)
    h2 = [0] * (N + 1)
    for x in range(1, N + 1):
        # random ints in [1..mod-1]
        h1[x] = random.randrange(1, mod1)
        h2[x] = random.randrange(1, mod2)
    
    # Build prefix hashes for A and B
    prefA1 = [0] * (N + 1)
    prefA2 = [0] * (N + 1)
    prefB1 = [0] * (N + 1)
    prefB2 = [0] * (N + 1)
    
    for i in range(1, N + 1):
        ai = A[i-1]
        bi = B[i-1]
        prefA1[i] = (prefA1[i-1] + h1[ai]) % mod1
        prefA2[i] = (prefA2[i-1] + h2[ai]) % mod2
        prefB1[i] = (prefB1[i-1] + h1[bi]) % mod1
        prefB2[i] = (prefB2[i-1] + h2[bi]) % mod2
    
    out = []
    append = out.append
    
    for _ in range(Q):
        l, r, L, R = map(int, data.readline().split())
        # Check equal length first
        if (r - l) != (R - L):
            append("No")
            continue
        # Compute the hash sums of the two subranges
        ha1 = prefA1[r] - prefA1[l-1]
        ha2 = prefA2[r] - prefA2[l-1]
        hb1 = prefB1[R] - prefB1[L-1]
        hb2 = prefB2[R] - prefB2[L-1]
        # Normalize to positive modulo
        ha1 %= mod1
        ha2 %= mod2
        hb1 %= mod1
        hb2 %= mod2
        
        if ha1 == hb1 and ha2 == hb2:
            append("Yes")
        else:
            append("No")
    
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()