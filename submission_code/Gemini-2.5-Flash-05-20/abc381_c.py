import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    # prefix_ones[i] stores the number of consecutive '1's immediately to the left of index i.
    # That is, it counts '1's in S[i-1], S[i-2], ... until a non-'1' character or start of string.
    prefix_ones = [0] * N
    for i in range(1, N):
        if S[i-1] == '1':
            prefix_ones[i] = prefix_ones[i-1] + 1
        else:
            prefix_ones[i] = 0

    # suffix_twos[i] stores the number of consecutive '2's immediately to the right of index i.
    # That is, it counts '2's in S[i+1], S[i+2], ... until a non-'2' character or end of string.
    suffix_twos = [0] * N
    for i in range(N - 2, -1, -1):
        if S[i+1] == '2':
            suffix_twos[i] = suffix_twos[i+1] + 1
        else:
            suffix_twos[i] = 0

    max_length = 0

    # Iterate through each character in S
    for i in range(N):
        if S[i] == '/':
            # If S[i] is '/', it can be the middle character of an 11/22 string.
            # The number of '1's to the left (k) must equal the number of '2's to the right (k).
            # We can form k ones and k twos, where k is the minimum of available counts.
            k = min(prefix_ones[i], suffix_twos[i])
            
            # The length of such an 11/22 string is k (ones) + 1 (slash) + k (twos) = 2*k + 1
            current_length = 2 * k + 1
            
            # Update the maximum length found so far
            max_length = max(max_length, current_length)
            
    # As per constraints, S contains at least one '/'.
    # So max_length will be at least 1 (for the '/' itself, when k=0).
    print(max_length)

solve()