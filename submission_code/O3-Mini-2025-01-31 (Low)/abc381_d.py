def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    # A 1122 sequence must have even length. It is formed by pairs of equal consecutive numbers.
    # In addition, if a number appears, it appears exactly once as a pair (i.e. exactly twice overall).
    # Hence, if we view the 1122 sequence as a sequence of pairs (x, x) arranged consecutively,
    # these pairs are contiguous in A. Moreover, the integer in one pair must be different than the integer in any other pair.
    #
    # Therefore, if a candidate subarray (of even length) starts at index L and ends at index R,
    # writing it as pairs we require for every k with L <= k <= R-1 stepping by 2:
    #   (A[k], A[k+1]) forms a valid pair (A[k] == A[k+1]),
    # and the number A[k] is not repeated in any other pair.
    #
    # We can take advantage of:
    #   - The contiguous pairs structure forces that the starting index L mod 2 remains constant.
    #   - For a fixed parity p (p = 0 or 1), consider positions: p, p+2, p+4, ... up to n-2.
    #     At each such index j, if A[j] == A[j+1] then (A[j], A[j+1]) is a valid pair.
    # 
    # However, if at some position j the pair condition fails (A[j] != A[j+1]),
    # then we cannot extend a candidate subarray across j. Thus the array splits into segments
    # where every potential pair (with the chosen parity) is valid.
    #
    # Now, in a contiguous valid segment of pairs (say, a list B of pair values),
    # we need to choose the longest contiguous subsegment in which all values are distinct.
    # Each distinct pair contributes 2 to the subarray length.
    
    # We process parity p=0 and p=1 separately.
    ans = 0  # maximum number of pairs 
    for p in [0, 1]:
        B = []  # list of the number forming the valid pair, for the current segment.
        j = p
        while j <= n - 2:
            if A[j] == A[j + 1]:
                B.append(A[j])
            else:
                if B:
                    candidate = find_max_distinct_block(B)
                    if candidate > ans:
                        ans = candidate
                    B = []
                # When pair condition fails, we simply do not add anything.
            j += 2
        if B:
            candidate = find_max_distinct_block(B)
            if candidate > ans:
                ans = candidate
    # Each pair contributes 2 numbers.
    sys.stdout.write(str(ans * 2))
    
def find_max_distinct_block(B):
    # Given a list B representing consecutive pairs (each element is the number repeated in the pair)
    # We want the length (count of pairs) of the longest contiguous sub-block of B where every pair number (element) is distinct.
    # This can be computed using a standard two-pointer sliding window.
    left = 0
    seen = {}
    max_len = 0
    for right, val in enumerate(B):
        if val in seen and seen[val] >= left:
            left = seen[val] + 1
        seen[val] = right
        window_len = right - left + 1
        if window_len > max_len:
            max_len = window_len
    return max_len

if __name__ == '__main__':
    main()