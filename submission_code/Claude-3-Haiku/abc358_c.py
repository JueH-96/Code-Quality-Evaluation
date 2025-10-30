from collections import defaultdict

def solve():
    N, M = map(int, input().split())
    stands = [input() for _ in range(N)]

    # Create a dictionary to store the stands that sell each flavor
    flavor_to_stands = defaultdict(list)
    for i, stand in enumerate(stands):
        for j, char in enumerate(stand):
            if char == 'o':
                flavor_to_stands[j+1].append(i+1)

    # Find the minimum number of stands needed to cover all flavors
    min_stands = 0
    visited = set()
    for flavor, stand_list in flavor_to_stands.items():
        if not visited.intersection(stand_list):
            min_stands += 1
            visited.update(stand_list)

    print(min_stands)

solve()