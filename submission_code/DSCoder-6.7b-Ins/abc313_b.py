import sys
from collections import defaultdict

def read_input():
    N, M = map(int, sys.stdin.readline().split())
    superiority = defaultdict(set)
    for _ in range(M):
        A, B = map(int, sys.stdin.readline().split())
        superiority[A].add(B)
    return N, M, superiority

def solve():
    N, M, superiority = read_input()
    strongest = -1
    for person in range(1, N+1):
        is_strongest = True
        for other_person in range(1, N+1):
            if other_person != person and (other_person not in superiority[person] or any(other_person not in superiority[sup] for sup in superiority[person])):
                is_strongest = False
                break
        if is_strongest:
            if strongest != -1:
                return -1
            strongest = person
    return strongest

print(solve())