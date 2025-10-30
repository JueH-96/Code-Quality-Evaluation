import sys

def solve():
    N, T, P = map(int, sys.stdin.readline().split())
    L = list(map(int, sys.stdin.readline().split()))

    # Sort the hair lengths in ascending order.
    # The P-th person with the longest hair will be at index N - P (0-based)
    # in the sorted list.
    L.sort()

    # The hair length of the P-th person with the longest hair (initially).
    p_th_longest_initial_length = L[N - P]

    # We need this person's hair length to be at least T.
    # Let d be the number of days. The length after d days is p_th_longest_initial_length + d.
    # We need p_th_longest_initial_length + d >= T.
    # This inequality implies d >= T - p_th_longest_initial_length.
    # The minimum integer value for d is T - p_th_longest_initial_length.

    # However, the number of days cannot be negative. If the P-th longest hair
    # is already >= T (i.e., T - p_th_longest_initial_length <= 0), then the condition
    # is already met on day 0, and the answer is 0.
    # So, the required number of days is max(0, T - p_th_longest_initial_length).
    required_days = max(0, T - p_th_longest_initial_length)

    print(required_days)

if __name__ == "__main__":
    solve()