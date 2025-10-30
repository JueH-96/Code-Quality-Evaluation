import sys

def read_input():
    N = int(sys.stdin.readline().strip())
    people = []
    for i in range(N):
        A, B = map(int, sys.stdin.readline().strip().split())
        people.append((i+1, A, A+B))
    return N, people

def solve():
    N, people = read_input()
    people.sort(key=lambda x: (-(x[1]/x[2]), x[0]))
    for person in people:
        print(person[0], end=' ')

solve()