# YOUR CODE HERE
import sys

def find_descendants(person, generations, descendants, family):
    if generations == 0:
        return

    for child in family[person]:
        descendants.add(child)
        find_descendants(child, generations - 1, descendants, family)

def main():
    N, M = map(int, input().split())
    parents = [0] + list(map(int, input().split()))

    family = [[] for _ in range(N + 1)]
    for i in range(2, N + 1):
        family[parents[i]].append(i)

    covered = set()
    for _ in range(M):
        x, y = map(int, input().split())
        descendants = set([x])
        find_descendants(x, y, descendants, family)
        covered.update(descendants)

    print(len(covered))

if __name__ == "__main__":
    main()