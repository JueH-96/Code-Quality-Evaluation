def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    S = data[1].strip()
    
    # Count digit frequencies of S (digits 0 through 9)
    freq_S = [0] * 10
    for ch in S:
        freq_S[ord(ch) - 48] += 1
    freq_S = tuple(freq_S)
    
    # The maximum number that can be obtained by any permutation
    # is the number formed by the digits sorted in descending order.
    max_val = int("".join(sorted(S, reverse=True)))
    # Compute the maximum k such that k*k is <= max_val.
    kmax = math.isqrt(max_val)
    
    found_squares = set()
    # Iterate k from 0 to kmax (inclusive). For each, check if its square,
    # when represented as an N-digit number (with leading zeros if needed),
    # has the same digit frequency as S.
    for k in range(kmax + 1):
        square = k * k
        # If the square has fewer than N digits, pad with leading zeros.
        s_square = str(square).zfill(N)
        # If even after padding the length is too long (should not happen),
        # skip by breaking out.
        if len(s_square) > N:
            break
        
        # Compute frequency for the candidate square.
        freq_candidate = [0] * 10
        for d in s_square:
            freq_candidate[ord(d) - 48] += 1
        if tuple(freq_candidate) == freq_S:
            found_squares.add(square)
    
    sys.stdout.write(str(len(found_squares)))

if __name__ == '__main__':
    main()