import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    count = 0

    while True:
        # Sort in descending order
        # This brings the largest elements to the front.
        A.sort(reverse=True)

        # Check termination condition: are there at least two positive elements?
        # The loop continues as long as there are two or more elements > 0.
        # After sorting, with N >= 2 (guaranteed by constraints),
        # there are at least two positive elements if and only if the second element A[1] is positive.
        # If A[1] <= 0, then A[2], A[3], ... are also <= 0 (due to sorting).
        # Thus, at most A[0] can be positive. This means there are 0 or 1 positive elements.
        if A[1] <= 0:
             break # Stop if the second element is not positive (implies <= 1 positive element)

        # Perform operation: Decrease the first two elements by 1
        # When the loop condition A[1] > 0 is met, both A[0] and A[1] must be positive
        # because A[0] >= A[1] due to sorting. So, decreasing them by 1 is valid.
        A[0] -= 1
        A[1] -= 1

        # Increment operation count
        count += 1

    # Print the result
    print(count)

solve()