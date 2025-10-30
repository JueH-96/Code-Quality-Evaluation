# YOUR CODE HERE
import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    parents = [0] + list(map(int, sys.stdin.readline().split()))
    children = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        children[parents[i-1]].append(i)
    
    insurances = []
    for _ in range(m):
        insurances.append(list(map(int, sys.stdin.readline().split())))
    
    covered = [False] * (n + 1)
    
    for x, y in insurances:
        q = [(x, 0)]
        covered_now = set()
        while q:
            person, gen = q.pop(0)
            if person not in covered_now:
                covered_now.add(person)
                if gen < y:
                    for child in children[person]:
                        q.append((child, gen + 1))

        for person in covered_now:
            covered[person] = True
            
    print(sum(covered))

solve()