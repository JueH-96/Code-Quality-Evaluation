from itertools import combinations

def min_stands_to_visit(N, M, stands):
    """
    Determine the minimum number of stands Takahashi needs to visit to buy all the flavors of popcorn.

    Args:
    N (int): The number of popcorn stands.
    M (int): The number of different flavors of popcorn.
    stands (list): A list of strings, where each string represents the flavors of popcorn sold at a stand.

    Returns:
    int: The minimum number of stands Takahashi needs to visit.
    """
    # Generate all possible combinations of stands
    for r in range(1, N + 1):
        for combo in combinations(range(N), r):
            # Check if the current combination of stands sells all flavors
            flavors_sold = set()
            for stand in combo:
                for j, flavor in enumerate(stands[stand]):
                    if flavor == 'o':
                        flavors_sold.add(j)
            if len(flavors_sold) == M:
                return r

# Read the input from stdin
N, M = map(int, input().split())
stands = [input() for _ in range(N)]

# Solve the problem and write the answer to stdout
print(min_stands_to_visit(N, M, stands))