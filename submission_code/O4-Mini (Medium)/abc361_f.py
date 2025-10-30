import sys
import threading

def main():
    import sys
    from math import isqrt
    
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    
    # We'll collect all "pure" powers a^b for b >= 3 in S_all,
    # and among them mark those that are perfect squares in S_sq.
    S_all = set()
    S_sq  = set()
    
    # Generate a^b for b>=3 by increasing base a.
    a = 2
    while True:
        # start with b = 3
        p = a * a * a
        if p > N:
            break
        b = 3
        # keep multiplying by a until we exceed N
        while p <= N:
            S_all.add(p)
            # if exponent b is even, p = (a^(b/2))^2 is a square
            if (b & 1) == 0:
                S_sq.add(p)
            p *= a
            b += 1
        a += 1
    
    # Count of perfect squares x = a^2 up to N, with a>=2:
    # that is floor(sqrt(N)) - 1 (we subtract a=1 to exclude 1^2=1 here)
    sq_count = isqrt(N) - 1
    if sq_count < 0:
        sq_count = 0
    
    # Among the b>=3 powers we generated, some are squares (we
    # have them in S_sq).  We want the count of non-square
    # higher powers:
    non_sq_higher = len(S_all) - len(S_sq)
    
    # Total distinct perfect powers in [1..N]:
    #   1 counts (as 1 = 1^b for b>=2),
    #   plus all distinct squares >=4,
    #   plus all distinct higher powers that aren't squares.
    ans = 1 + sq_count + non_sq_higher
    
    print(ans)

if __name__ == "__main__":
    main()