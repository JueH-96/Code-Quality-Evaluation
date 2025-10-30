# YOUR CODE HERE
from itertools import combinations

def is_valid_partition(partition, incompatible_pairs):
    for team in partition:
        for a, b in incompatible_pairs:
            if a in team and b in team:
                return False
    return True

def count_partitions(players, T, incompatible_pairs):
    if T == 1:
        return 1 if is_valid_partition([players], incompatible_pairs) else 0
    
    count = 0
    for comb in combinations(players, T - 1):
        remaining = set(players) - set(comb)
        sub_partitions = count_partitions(list(remaining), T - 1, incompatible_pairs)
        count += sub_partitions
    return count

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    M = int(data[2])
    
    incompatible_pairs = []
    for i in range(M):
        A = int(data[3 + 2 * i])
        B = int(data[4 + 2 * i])
        incompatible_pairs.append((A, B))
    
    players = list(range(1, N + 1))
    result = count_partitions(players, T, incompatible_pairs)
    print(result)

if __name__ == "__main__":
    main()