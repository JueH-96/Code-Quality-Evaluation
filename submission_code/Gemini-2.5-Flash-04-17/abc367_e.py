import sys

# Read N and K
line1 = sys.stdin.readline().split()
N = int(line1[0])
K = int(line1[1])

# Read X and convert to 0-based mapping
# mapping[i] = index in the previous array whose value moves to index i
# i.e., B[i] = A[X[i]-1]. So mapping[i] = X[i]-1.
X_input = list(map(int, sys.stdin.readline().split()))
mapping = [x - 1 for x in X_input]

# Read A (the initial sequence)
A = list(map(int, sys.stdin.readline().split()))

# If K is 0, no operations are performed
if K == 0:
    # Print A space-separated
    print(*A)
else:
    # Binary exponentiation to compute mapping^K(i) for each i
    # result_indices[i] stores the index in the original array A that
    # contributes to the value at index i after the transformation.
    # Initially, result_indices[i] = i (identity transformation after 0 steps)
    result_indices = list(range(N))

    # transform[i] stores the result of applying the basic mapping function
    # some power of 2 number of times.
    # Initially, transform represents mapping^1.
    transform = list(mapping)

    # Use binary representation of K
    # We iterate while K > 0. In each iteration, we handle one bit of the original K.
    # The current `transform` represents mapping^{2^j} where j is the current bit position (LSB).
    # The `result_indices` represents mapping^k where k is the sum of 2^b for bits b < j that were 1.
    while K > 0:
        # If the current bit (LSB of K) is 1, apply the current power of transformation
        # to the accumulated result.
        # If result_indices represents function f and transform represents function g,
        # the new result should be g(f(i)).
        if K % 2 == 1:
            next_result_indices = [0] * N
            for i in range(N):
                next_result_indices[i] = transform[result_indices[i]]
            result_indices = next_result_indices # Update result_indices

        # Square the transformation function for the next bit
        # If transform represents function g, the new transform represents g o g.
        # (g o g)(i) = g(g(i)) = transform[transform[i]]
        next_transform = [0] * N
        for i in range(N):
            next_transform[i] = transform[transform[i]]
        transform = next_transform # Update transform

        # Move to the next bit of K
        K //= 2

    # After the loop, result_indices[i] holds mapping^K(i).
    # The final sequence A_prime has A_prime[i] = A_initial[mapping^K(i)].
    A_prime = [A[result_indices[i]] for i in range(N)]

    # Print A_prime space-separated
    print(*A_prime)