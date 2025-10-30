import sys
from collections import defaultdict

def main():
    N, M = map(int, sys.stdin.readline().split())
    parents = [0] + list(map(int, sys.stdin.readline().split()))
    insurances = [tuple(map(int, sys.stdin.readline().split())) for _ in range(M)]

    # Create a dictionary to store the descendants of each person
    descendants = defaultdict(set)
    for i in range(2, N+1):
        person = i
        while person != 1:
            descendants[parents[person]].add(i)
            person = parents[person]

    # Check which people are covered by each insurance
    covered = set()
    for x, y in insurances:
        for generation in range(y):
            for person in list(descendants[x]):
                covered.add(person)
            descendants[x] = set()

    # Print the number of people covered by at least one insurance
    print(len(covered))

if __name__ == "__main__":
    main()