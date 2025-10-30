from itertools import permutations, combinations

def count_ways(N, T, M, pairs):
    def is_valid(team, pairs):
        for a, b in pairs:
            if team[a-1] == team[b-1]:
                return False
        return True

    def count_partitions(n, k):
        if n == 0 and k == 0:
            return 1
        if n == 0 or k == 0:
            return 0
        return count_partitions(n-1, k-1) + k * count_partitions(n-1, k)

    def generate_partitions(n, k):
        if n == 0 and k == 0:
            yield []
        elif n == 0 or k == 0:
            return
        else:
            for partition in generate_partitions(n-1, k-1):
                yield [1] + [x+1 for x in partition]
            for partition in generate_partitions(n-1, k):
                for i in range(k):
                    yield [i+1] + [x for x in partition]

    ways = 0
    for partition in generate_partitions(N, T):
        if is_valid(partition, pairs):
            ways += 1

    return ways

N, T, M = map(int, input().split())
pairs = [tuple(map(int, input().split())) for _ in range(M)]

print(count_ways(N, T, M, pairs))