from heapq import heappop, heappush

def solve_building_visibility(heights):
    """
    Solves the problem of determining the visibility of buildings based on their heights.
    For each building, counts how many buildings are visible to its right until an obstacle is found.
    """
    if not heights:
        return []

    visibility = []
    sorted_buildings = []
    for index, height in enumerate(reversed(heights)):
        while sorted_buildings and sorted_buildings[0][0] < height:
            heappop(sorted_buildings)
        visibility.append(index - len(sorted_buildings))
        heappush(sorted_buildings, (height, -index))

    return list(reversed(visibility))

# Reading input from stdin
N = int(input())
heights = list(map(int, input().split()))

# Solving the problem
result = solve_building_visibility(heights)

# Writing output to stdout
print(' '.join(map(str, result)))