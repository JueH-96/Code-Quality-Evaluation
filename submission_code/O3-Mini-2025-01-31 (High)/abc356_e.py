def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Read input.
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    max_val = max(A)
    
    # Build a frequency array for values 1..max_val.
    freq = [0] * (max_val + 1)
    for a in A:
        freq[a] += 1

    # Build a prefix sum array: prefix[i] = freq[1] + freq[2] + ... + freq[i].
    prefix = [0] * (max_val + 1)
    s = 0
    for i in range(1, max_val + 1):
        s += freq[i]
        prefix[i] = s

    ans = 0
    # For each value u that appears in A, we will count pairs (u,v)
    # where u is the smaller element. If both are equal, floor(u/u)=1.
    for u in range(1, max_val + 1):
        if freq[u] == 0:
            continue
        fu = freq[u]

        # Count pairs with both values equal to u.
        if fu > 1:
            ans += fu * (fu - 1) // 2  # floor(u/u) is 1.

        # Now count pairs where the smaller element is u and the larger v > u.
        # For any such pair, with u as smaller and v as larger, we have
        # floor(v/u) = q if and only if v is in [q*u, (q+1)*u - 1].
        # We iterate over possible q.
        limit = max_val // u
        for q in range(1, limit + 1):
            L = q * u
            R = (q + 1) * u - 1
            if R > max_val:
                R = max_val
            # Count how many numbers v are in the interval [L, R].
            cnt = prefix[R] - (prefix[L - 1] if L > 1 else 0)
            # For q==1 the interval [u, 2*u - 1] includes u itself.
            # We already counted pairs with both numbers equal to u.
            if q == 1:
                cnt -= fu
            if cnt <= 0:
                continue
            # For each occurrence of u, there are cnt numbers v
            # and each pair contributes q.
            ans += fu * cnt * q

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()