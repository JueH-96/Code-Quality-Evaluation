# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import bisect
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    MAX_A = 2 * 10**5 + 5
    earliest_person_with_Ai_leq_A = [N + 1] * (MAX_A + 2)

    for i in range(N -1, -1, -1):
        Ai = A[i]
        earliest_person_with_Ai_leq_A[Ai] = i +1  # Positions are from 1, not 0.

    for A_level in range(1, MAX_A +1):
        earliest_person_with_Ai_leq_A[A_level] = min(
            earliest_person_with_Ai_leq_A[A_level],
            earliest_person_with_Ai_leq_A[A_level -1]
        )

    for B_j in B:
        earliest_person = earliest_person_with_Ai_leq_A[B_j]
        if earliest_person <= N:
            print(earliest_person)
        else:
            print(-1)