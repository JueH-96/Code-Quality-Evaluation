import sys

def solve():
    # Read N, L, R from standard input
    N, L, R = map(int, sys.stdin.readline().split())

    total_sum = 0
    l = L
    r = R + 1  # Use half-open interval [l, r) for decomposition
    p = 0      # Current power of 2 for block size is 2^p

    while l < r:
        if l % 2 == 1:
            # Current effective start index l * 2^p corresponds to the start of a block of size 2^p.
            # Since l is odd, this block starts at an odd multiple of 2^p.
            # This block in original indices is [l * 2^p, (l+1) * 2^p - 1].
            # This corresponds to a query with i = p and j = l.
            # Constraints for query ? i j: i >= 0, j >= 0, 2^i * (j+1) <= 2^N.
            # i = p >= 0 is true as p starts at 0 and increases.
            # j = l >= 0 is true as l starts at L >= 0 and only increases or is halved from non-negative values.
            # 2^p * (l+1) <= 2^N. At any stage, the range [l * 2^p, r * 2^p) is a sub-interval of [L, R+1).
            # Thus, r * 2^p <= R+1 <= 2^N, which implies r <= 2^(N-p).
            # Since l < r and l is odd, l+1 <= r. So l+1 <= 2^(N-p), which implies 2^p * (l+1) <= 2^N.
            # This confirms the query ? p l is valid.
            print(f"? {p} {l}")
            sys.stdout.flush()
            T = int(sys.stdin.readline())
            if T == -1:
                exit() # Judge returned error, terminate
            total_sum = (total_sum + T) % 100
            l += 1 # Move the left boundary past the block just covered

        elif r % 2 == 1:
            # Current effective end index r * 2^p corresponds to the end of a block of size 2^p.
            # Since r is odd, r-1 is even. This block starts at an even multiple of 2^p, which is (r-1)*2^p.
            # The block in original indices is [(r-1) * 2^p, r * 2^p - 1].
            # This corresponds to a query with i = p and j = r-1.
            # Constraints: i = p >= 0 (true). j = r-1 >= 0. r starts at R+1 >= 1. r only decreases or is halved.
            # If r=1 at level p, then l must be 0 (since l < r). j=r-1=0. Valid.
            # If r >= 2, r-1 >= 1 >= 0. j >= 0 is true.
            # 2^i * (j+1) <= 2^N, i.e., 2^p * ((r-1)+1) <= 2^N => 2^p * r <= 2^N.
            # As shown above, r * 2^p <= R+1 <= 2^N, so r <= 2^(N-p).
            # This confirms the query ? p (r-1) is valid.
            print(f"? {p} {r-1}")
            sys.stdout.flush()
            T = int(sys.stdin.readline())
            if T == -1:
                exit() # Judge returned error, terminate
            total_sum = (total_sum + T) % 100
            r -= 1 # Move the right boundary before the block just covered

        else:
            # Both l and r are even. The current range [l * 2^p, r * 2^p) can be expressed
            # as [ (l/2) * 2^(p+1), (r/2) * 2^(p+1) ).
            # Scale down the problem to the next level by dividing l and r by 2 and increasing the power p.
            l //= 2
            r //= 2
            p += 1 # Increase the power of 2 for the block size

    # Output the final remainder
    print(f"! {total_sum}")
    sys.stdout.flush()

solve()