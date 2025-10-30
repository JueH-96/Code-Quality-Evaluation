def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Read N and Q.
    n = int(data[0])
    q = int(data[1])
    # The bit‐string S is given in one token (as bytes).
    # Each byte is the ascii code for '0' or '1'. Subtract 48 to get 0/1.
    s_bytes = data[2]
    A = [b - 48 for b in s_bytes]

    # We want to support two types of queries on S.
    # A “good string” is one with no two equal adjacent digits.
    # We will maintain an auxiliary “diff” array of length n-1,
    # where for 0 <= i < n-1, diff[i] = 1 if S[i] == S[i+1], and 0 otherwise.
    # Then a query of type 2 (check) on a substring S[L..R] (1-indexed)
    # is “Yes” if the sum of diff over indices L-1 to R-2 is 0.
    #
    # However, lots of queries include “flip” operations (changing 0→1 and 1→0)
    # on a segment. Notice that flipping an entire contiguous segment does NOT change
    # the relationship of pairs completely inside the segment (since both are flipped)
    # – only the boundaries are affected.
    #
    # To support many queries (up to 5×10^5) with range flips we use two Fenwick trees:
    #
    # (1) BIT_flip: maintains range‐flip updates (each flip is “xor 1”) so the current
    #     value of S[i] is A[i] XOR (flip parity at i). (BIT_flip uses XOR as “addition”.)
    #
    # (2) BIT_diff: a standard BIT (Fenwick tree) to hold the diff array (storing sums).
    #
    # In a flip query “1 L R” (1-indexed, meaning flip S[L..R]), only the diff at positions L-1 and R might change.
    #
    # We now set up our BITs.
    
    # BIT_flip: size n+1, 1-indexed.
    bit_flip = [0] * (n + 1)
    def f_update(i, delta):
        while i <= n:
            bit_flip[i] ^= delta
            i += i & -i
    def f_query(i):
        s_val = 0
        while i:
            s_val ^= bit_flip[i]
            i -= i & -i
        return s_val

    # BIT_diff: will be built over diff[0..n-2]. We use indices 1..n-1 (1-indexed) for BIT_diff.
    bit_diff = [0] * n  # note: we use indices 1..(n-1)
    def d_update(i, delta):
        # i is 1-indexed.
        while i < n:
            bit_diff[i] += delta
            i += i & -i
    def d_query(i):
        s_sum = 0
        while i:
            s_sum += bit_diff[i]
            i -= i & -i
        return s_sum
    def d_range_query(l, r):
        return d_query(r) - d_query(l - 1)
    
    # Build the initial diff array (for positions 0 to n-2) and update BIT_diff.
    diff = [0] * (n - 1)
    for i in range(n - 1):
        if A[i] == A[i + 1]:
            diff[i] = 1
            j = i + 1
            while j < n:
                bit_diff[j] += 1
                j += j & -j

    # We'll accumulate our answers for type 2 queries here.
    out_lines = []
    # Queries start at data index 3.
    pos = 3
    for _ in range(q):
        typ = int(data[pos]); pos += 1
        L = int(data[pos]); pos += 1
        R = int(data[pos]); pos += 1
        if typ == 1:
            # Query type 1: flip S[L..R]. (Indices L-1..R-1 in A.)
            # Do a range update on BIT_flip: add 1 (via XOR) to positions L through R.
            f_update(L, 1)
            if R + 1 <= n:
                f_update(R + 1, 1)
            # Only the boundaries of the flipped segment affect the "diff" array.
            # Left boundary: if L > 1, update diff at index L-2 (comparing A[L-2] and A[L-1]).
            if L > 1:
                i_diff = L - 2  # 0-indexed diff index.
                # New current values at positions (L-2) and (L-1):
                s_left = A[L - 2] ^ f_query(L - 1)
                s_right = A[L - 1] ^ f_query(L)
                new_val = 1 if s_left == s_right else 0
                if new_val != diff[i_diff]:
                    d_update(L - 1, new_val - diff[i_diff])
                    diff[i_diff] = new_val
            # Right boundary: if R < n, update diff at index R-1 (comparing A[R-1] and A[R]).
            if R < n:
                i_diff = R - 1
                s_left = A[R - 1] ^ f_query(R)
                s_right = A[R] ^ f_query(R + 1)
                new_val = 1 if s_left == s_right else 0
                if new_val != diff[i_diff]:
                    d_update(R, new_val - diff[i_diff])
                    diff[i_diff] = new_val
        else:
            # Query type 2: Check if S[L..R] is good.
            # A single‐character string always is good.
            if L == R:
                out_lines.append("Yes")
            else:
                # For S[L..R] (1-indexed), the relevant diff indices are from L-1 to R-2 (0‐indexed).
                # In BIT_diff, these correspond to indices L to R-1 (1-indexed).
                if d_range_query(L, R - 1) == 0:
                    out_lines.append("Yes")
                else:
                    out_lines.append("No")
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()