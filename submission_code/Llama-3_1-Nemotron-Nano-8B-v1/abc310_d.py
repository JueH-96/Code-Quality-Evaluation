import sys

def main():
    N, T, M = map(int, sys.stdin.readline().split())
    incompatible = set()
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        incompatible.add((a, b))
    
    all_partitions = []
    
    def generate_partitions(current_player, current_subsets):
        if current_player > N:
            if len(current_subsets) == T:
                all_partitions.append(current_subsets.copy())
            return
        for i in range(len(current_subsets)):
            new_subsets = [s.copy() for s in current_subsets]
            new_subsets[i].append(current_player)
            generate_partitions(current_player + 1, new_subsets)
        if len(current_subsets) < T:
            new_subsets = [s.copy() for s in current_subsets]
            new_subsets.append([current_player])
            generate_partitions(current_player + 1, new_subsets)
    
    generate_partitions(2, [[1]])
    
    count = 0
    for partition in all_partitions:
        player_to_subset = {}
        for i, subset in enumerate(partition):
            for player in subset:
                player_to_subset[player] = i
        valid = True
        for a, b in incompatible:
            if player_to_subset.get(a, -1) == player_to_subset.get(b, -2):
                valid = False
                break
        if valid:
            count += 1
    print(count)

if __name__ == '__main__':
    main()