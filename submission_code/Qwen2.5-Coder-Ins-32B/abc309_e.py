# YOUR CODE HERE
import sys
import collections

def main():
    import sys
    import collections

    def input():
        return sys.stdin.read().split()

    data = input()
    index = 0
    N = int(data[index])
    M = int(data[index + 1])
    index += 2

    parents = [0] * (N + 1)
    for i in range(2, N + 1):
        parents[i] = int(data[index])
        index += 1

    children = collections.defaultdict(list)
    for i in range(2, N + 1):
        children[parents[i]].append(i)

    covered = set()

    for _ in range(M):
        x = int(data[index])
        y = int(data[index + 1])
        index += 2

        queue = collections.deque([(x, 0)])
        while queue:
            person, gen = queue.popleft()
            if person not in covered and gen <= y:
                covered.add(person)
                for child in children[person]:
                    queue.append((child, gen + 1))

    print(len(covered))

if __name__ == "__main__":
    main()