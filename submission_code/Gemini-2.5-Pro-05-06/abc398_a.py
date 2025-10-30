# YOUR CODE HERE
N = int(input())

if N % 2 == 1:
    # N is odd: The string must have exactly one '=' at the center.
    # The number of '-' characters on each side of the central '=' is (N-1)/2.
    num_dashes_each_side = (N - 1) // 2
    result = "-" * num_dashes_each_side + "=" + "-" * num_dashes_each_side
    print(result)
else:
    # N is even: The string must have exactly two adjacent '='s ("==") at the center.
    # The number of '-' characters on each side of the central "==" is (N-2)/2, which is N/2 - 1.
    num_dashes_each_side = (N // 2) - 1
    result = "-" * num_dashes_each_side + "==" + "-" * num_dashes_each_side
    print(result)