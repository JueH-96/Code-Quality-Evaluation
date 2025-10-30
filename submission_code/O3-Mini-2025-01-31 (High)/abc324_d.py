import sys, math

def main():
    # Read all input and parse N and S.
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    S = data[1].strip()
    
    # We can represent the multiset (frequency) of the digits in S
    # by sorting S. (Order does not matter, we only want to recognize if
    # two numbers have the same digits with the same frequencies.)
    sortedS = "".join(sorted(S))
    
    # Precompute a string of N zeros for fast padding.
    zeros = "0" * N
    
    # Every permutation of S is an arrangement of exactly N digits.
    # In particular, even if the number normally has fewer than N digits,
    # we want to think of it as its N-digit version (with leading zeros.)
    # Therefore, we are looking for perfect squares less than 10^N.
    U = 10 ** N
    # Let max_k be the largest integer such that k*k < 10^N.
    max_k = math.isqrt(U - 1)
    
    ans = 0
    # For speed, cache local names for built‐in functions.
    join_func = "".join
    sorted_func = sorted
    
    # Iterate over all possible k with k*k < 10^N.
    for k in range(max_k + 1):
        sq = k * k
        # Convert the square to string.
        cand = str(sq)
        # If the number has less than N digits, pad it with leading zeros.
        if len(cand) < N:
            cand = zeros[:N - len(cand)] + cand
        # Check if this candidate, when its digits are re‐ordered (via sorting),
        # gives the same multiset as S.
        if join_func(sorted_func(cand)) == sortedS:
            ans += 1

    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()