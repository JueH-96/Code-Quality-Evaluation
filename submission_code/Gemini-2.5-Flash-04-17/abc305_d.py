import sys
from bisect import bisect_right

# Use fast I/O
input = sys.stdin.readline

# Function to calculate total sleep time from A[0]=0 up to time t
def get_total_sleep_up_to(t, A, S_prefix):
    """
    Calculates the total sleep time from time A[0]=0 up to time t.
    A: The list of time points (0-indexed from input A_1 to A_N)
    S_prefix: Precomputed total sleep time up to each A[i]
    """
    # Find the index `idx` in A such that A[idx-1] <= t < A[idx].
    # bisect_right finds the insertion point `idx` for `t` in `A`
    # such that all elements A[0...idx-1] are <= t, and all elements A[idx...N-1] are > t.
    idx = bisect_right(A, t)

    # The time t falls into the interval [A[idx-1], A[idx]).
    # The relevant index k for determining the status in this interval is idx-1.
    # If t is exactly A[k'], bisect_right returns k'+1, so idx = k'+1, k = k'.
    # A[k] is the largest time point in A that is less than or equal to t.
    k = idx - 1

    # If k < 0, it means t < A[0]. Since A[0]=0 and t>=0 by constraints,
    # this happens only if t=0 and A[0]=0 (bisect_right returns 1, k=0)
    # or hypothetically if t<0 (not allowed) or A[0]>0 (not allowed).
    # For t=0, k=0, S_prefix[0]=0, returned value is S_prefix[0] + 0 (k=0 even) = 0. Correct.
    if k < 0:
         return 0

    # Total sleep accumulated exactly up to time A[k] is S_prefix[k].
    total_sleep = S_prefix[k]

    # The interval [A[k], A[k+1]) (or [A[k], infinity) if k=N-1) determines the sleep status between A[k] and A[k+1].
    # A[k] is a time point from the input list.
    # If k is odd (1, 3, 5, ...), A[k] is a bed time. The interval [A[k], A[k+1]) is a sleep period.
    # If k is even (0, 2, 4, ...), A[k] is a wake up time. The interval [A[k], A[k+1]) is an awake period.

    # If the interval [A[k], t] is part of a sleep period (k is odd), add the duration (t - A[k]).
    if k % 2 == 1:
        # The time elapsed in the current interval [A[k], t] is t - A[k].
        # Since k is odd, this interval corresponds to a sleep session.
        total_sleep += (t - A[k])

    # If k is even, the interval [A[k], t] is part of an awake period, so no additional sleep time.
    return total_sleep

# Read N
N = int(input())
# Read A, convert to a list of integers. This corresponds to A_1...A_N
A = list(map(int, input().split()))

# Calculate prefix sums of total sleep duration up to each time point A[i].
# S_prefix[i] stores the total sleep time from A[0]=0 up to time A[i].
S_prefix = [0] * N
# S_prefix[0] = 0 (total sleep up to time A[0]=0 is 0)
for i in range(1, N):
    # A[i] is the i-th time point in the 0-indexed list.
    # If i is odd (1, 3, ...), A[i] is a bed time (corresponds to A_{i+1} in 1-based problem statement).
    # The interval [A[i-1], A[i]] is an awake period (from wake up at A[i-1] to bed at A[i]).
    # Total sleep up to A[i] is the same as total sleep up to A[i-1].
    if i % 2 == 1:
        S_prefix[i] = S_prefix[i-1]
    # If i is even (2, 4, ...), A[i] is a wake up time (corresponds to A_{i+1} in 1-based problem statement).
    # The interval [A[i-1], A[i]] is a sleep period (from bed at A[i-1] to wake up at A[i]).
    # Total sleep up to A[i] is total sleep up to A[i-1] plus the duration of this sleep period.
    else: # i % 2 == 0
        S_prefix[i] = S_prefix[i-1] + (A[i] - A[i-1])

# Read Q
Q = int(input())

# Process each query [l, r]
for _ in range(Q):
    l, r = map(int, input().split())

    # The total sleep time during the interval [l, r] is calculated as
    # (Total sleep time from 0 to r) - (Total sleep time from 0 to l).
    # This is equivalent to S(r) - S(l).
    sleep_until_r = get_total_sleep_up_to(r, A, S_prefix)
    sleep_until_l = get_total_sleep_up_to(l, A, S_prefix)

    print(sleep_until_r - sleep_until_l)