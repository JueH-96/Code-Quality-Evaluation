# YOUR CODE HERE
def solve():
    N = int(input())
    A = list(map(int, input().split()))

    # A sequence is strictly increasing if A[i] < A[i+1] for all valid i.
    # In 0-indexed Python lists, this means A[i] < A[i+1] for i from 0 to N-2.
    # The loop should iterate N-1 times for N-1 comparisons.
    # range(N-1) generates integers from 0 up to N-2.

    # Constraints: 2 <= N <= 100. So N-1 >= 1.
    # The generator expression will produce at least one boolean value.

    # The all() function returns True if all elements of the iterable are true.
    # It short-circuits, meaning it stops processing as soon as a False element is found.
    # This is efficient, as we only need one counterexample (A[i] >= A[i+1])
    # to determine the sequence is not strictly increasing.
    
    is_strictly_increasing = all(A[i] < A[i+1] for i in range(N - 1))

    if is_strictly_increasing:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()