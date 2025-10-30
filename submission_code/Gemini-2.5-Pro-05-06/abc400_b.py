import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())

    LIMIT = 10**9

    if N == 1:
        # X = sum_{i=0 to M} 1^i = M + 1
        # Since M <= 100, M+1 <= 101. This is always <= 10^9.
        print(M + 1)
        return

    current_sum = 0
    current_term = 1  # Represents N^i, starts with N^0 = 1

    # Loop for i from 0 to M. This loop runs M+1 times.
    for i in range(M + 1):
        # In this iteration, current_term holds N^i.
        # current_sum holds the sum of terms N^0 through N^(i-1).
        
        # Check if adding current_term (N^i) would make current_sum exceed LIMIT.
        # Condition: current_sum_old + N^i > LIMIT  <=>  N^i > LIMIT - current_sum_old
        # current_sum is sum of non-negative terms, and N^i is positive.
        # current_sum has been kept <= LIMIT in previous steps, so LIMIT - current_sum >= 0.
        if current_term > LIMIT - current_sum:
            print("inf")
            return
        
        current_sum += current_term
        # Now current_sum = (sum of N^0...N^(i-1)) + N^i.
        # And current_sum is guaranteed to be <= LIMIT.

        if i < M:
            # If this is not the last term (i.e., we haven't processed N^M yet),
            # prepare N^(i+1) for the next iteration by N^i * N.
            current_term *= N
            # Python's arbitrary precision integers handle large values of current_term.
            # The check at the start of the next iteration will determine if this new
            # large term causes the sum to exceed LIMIT.
            
    # If the loop completes, current_sum holds sum_{i=0 to M} N^i, and it's <= LIMIT.
    print(current_sum)

if __name__ == '__main__':
    solve()