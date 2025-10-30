# YOUR CODE HERE
import sys

# Increase recursion depth - not necessary for this iterative solution, but often included
# sys.setrecursionlimit(2000) # Default recursion depth is usually high enough for this problem structure.

def solve():
    # Read input N and K
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    K = int(line1[1])

    # Read input array A
    A = list(map(int, sys.stdin.readline().split()))

    # Maximum possible value for A_i is 10^6
    MAX_VAL = 1000000

    # freq[x]: stores the number of times value x appears in A
    freq = [0] * (MAX_VAL + 1)
    # pos[x]: stores a list of indices i where A[i] is equal to x
    pos = [[] for _ in range(MAX_VAL + 1)]

    # Populate freq and pos arrays
    for i in range(N):
        freq[A[i]] += 1
        pos[A[i]].append(i)

    # counts[g]: stores the total number of elements in A that are divisible by g
    counts = [0] * (MAX_VAL + 1)
    # Iterate through all possible GCD values g
    for g in range(1, MAX_VAL + 1):
        # Iterate through all multiples m of g
        for m in range(g, MAX_VAL + 1, g):
            # If multiple m is present in the input array A (i.e., appears at least once)
            if freq[m] > 0:
                # Add the frequency of m to counts[g]
                # This correctly counts how many elements in A are divisible by g
                counts[g] += freq[m]

    # answer[i]: stores the maximum possible GCD for A[i]
    answer = [0] * N
    # is_answered[i]: boolean flag indicating if the answer for A[i] has been found
    is_answered = [False] * N
    # Counter for how many answers have been found
    num_answered = 0

    # Iterate through possible GCD values g downwards from MAX_VAL to 1
    for g in range(MAX_VAL, 0, -1):
        # If we already found answers for all N elements, we can stop early
        if num_answered == N:
            break

        # Condition Check: Can g be the GCD of a K-subset including A_i?
        # For g to be a possible GCD of a K-subset including A_i (where g | A_i),
        # we need g | A_i AND there must be at least K-1 *other* elements A_j (j!=i)
        # such that g | A_j.
        # The total number of elements A_k divisible by g is counts[g].
        # If g | A_i, then A_i is one of these counts[g] elements.
        # The number of *other* elements divisible by g is counts[g] - 1.
        # We need counts[g] - 1 >= K - 1, which simplifies to counts[g] >= K.
        if counts[g] >= K:
            # If counts[g] >= K, then g is a candidate GCD for any A_i divisible by g.
            # Iterate through all multiples m of g
            for m in range(g, MAX_VAL + 1, g):
                # If the multiple m is present in the input array A
                if freq[m] > 0:
                    # For each index i where A[i] is equal to m (which is a multiple of g)
                    for i in pos[m]:
                        # If the answer for this specific index i hasn't been found yet
                        # Since we iterate g downwards, the first time we find a valid g
                        # for an index i, it will be the maximum such g.
                        if not is_answered[i]:
                            # Set the answer for index i to g
                            answer[i] = g
                            # Mark index i as answered
                            is_answered[i] = True
                            # Increment the counter for answered elements
                            num_answered += 1

    # Print the answers for each A_i
    for i in range(N):
        sys.stdout.write(str(answer[i]) + "
")

# Run the solve function
solve()