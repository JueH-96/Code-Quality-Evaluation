import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    cakes = []
    for _ in range(N):
        cakes.append(list(map(int, sys.stdin.readline().split())))

    max_total_value = 0

    # Iterate through all 8 combinations of signs for X, Y, Z
    signs = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
    ]

    for s1, s2, s3 in signs:
        # Calculate the value for each cake based on the current sign combination
        # We store a tuple (value, original_index) to allow sorting and
        # retrieve original cake attributes if needed, though here we only sum the values.
        cake_values = []
        for i in range(N):
            value = s1 * cakes[i][0] + s2 * cakes[i][1] + s3 * cakes[i][2]
            cake_values.append((value, i))

        # Sort cakes by their calculated value in descending order
        cake_values.sort(key=lambda x: x[0], reverse=True)

        # Select the top 2K cakes. The problem states we pair K pairs,
        # which requires selecting 2K distinct cakes.
        # The hypothesis is that the set of 2K cakes that maximizes the total price
        # is one of the sets formed by picking the top 2K cakes according to
        # some linear combination s1*X + s2*Y + s3*Z.
        # The total price for this chosen set is then the sum of these linear
        # combinations for the selected cakes.
        current_sum_of_values = 0
        for i in range(2 * K):
            current_sum_of_values += cake_values[i][0]

        # The maximum total price is the maximum sum of values found
        # across all 8 sign combinations.
        max_total_value = max(max_total_value, current_sum_of_values)

    print(max_total_value)

T = int(sys.stdin.readline())
for _ in range(T):
    solve()