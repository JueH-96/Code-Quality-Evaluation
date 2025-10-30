import sys
from sys import stdin

def main():
    N, X = map(int, stdin.readline().split())
    A = list(map(int, stdin.readline().split()))
    B = list(map(int, stdin.readline().split()))
    P = list(map(int, stdin.readline().split()))
    Q = list(map(int, stdin.readline().split()))

    # Find cycles in P
    visited_P = [False] * (N + 1)
    cycles_P = []
    for i in range(1, N + 1):
        if not visited_P[i]:
            cycle = []
            j = i
            while not visited_P[j]:
                visited_P[j] = True
                cycle.append(j)
                j = P[j - 1]
            cycles_P.append(cycle)

    # Find cycles in Q
    visited_Q = [False] * (N + 1)
    cycles_Q = []
    for i in range(1, N + 1):
        if not visited_Q[i]:
            cycle = []
            j = i
            while not visited_Q[j]:
                visited_Q[j] = True
                cycle.append(j)
                j = Q[j - 1]
            cycles_Q.append(cycle)

    # Check if all A[i] > 0 are in the same cycle as X in P
    in_cycle_P = [False] * (N + 1)
    for cycle in cycles_P:
        if X in cycle:
            for i in cycle:
                in_cycle_P[i] = True
    for i in range(1, N + 1):
        if A[i - 1] > 0 and not in_cycle_P[i]:
            print(-1)
            return

    # Check if all B[i] > 0 are in the same cycle as X in Q
    in_cycle_Q = [False] * (N + 1)
    for cycle in cycles_Q:
        if X in cycle:
            for i in cycle:
                in_cycle_Q[i] = True
    for i in range(1, N + 1):
        if B[i - 1] > 0 and not in_cycle_Q[i]:
            print(-1)
            return

    # Collect required nodes for red
    R = set()
    for cycle in cycles_P:
        if X in cycle:
            sum_a = sum(A[i - 1] for i in cycle)
            if sum_a > 0:
                for i in cycle:
                    if i != X:
                        R.add(i)

    # Collect required nodes for blue
    B_set = set()
    for cycle in cycles_Q:
        if X in cycle:
            sum_b = sum(B[i - 1] for i in cycle)
            if sum_b > 0:
                for i in cycle:
                    if i != X:
                        B_set.add(i)

    # The answer is the size of the union
    print(len(R.union(B_set)))

if __name__ == "__main__":
    main()