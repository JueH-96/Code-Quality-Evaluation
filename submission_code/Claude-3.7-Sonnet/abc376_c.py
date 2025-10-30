def solve():
    n = int(input())
    toy_sizes = list(map(int, input().split()))
    box_sizes = list(map(int, input().split()))
    
    # Sort toys in non-decreasing order of size
    toy_sizes.sort()
    
    # For each toy (from smallest to largest), check if it can be left out
    for idx, toy_size in enumerate(toy_sizes):
        other_toys = toy_sizes[:idx] + toy_sizes[idx+1:]
        if can_place_toys(other_toys, box_sizes):
            return toy_size
    
    return -1

def can_place_toys(toy_sizes, box_sizes):
    # Create a bipartite graph where an edge (toy, box) exists if the toy can fit in the box
    toy_to_boxes = [[] for _ in range(len(toy_sizes))]
    for i, toy_size in enumerate(toy_sizes):
        for j, box_size in enumerate(box_sizes):
            if toy_size <= box_size:
                toy_to_boxes[i].append(j)
    
    # Find the maximum bipartite matching
    box_to_toy = [-1] * len(box_sizes)
    
    def dfs(toy, visited):
        for box in toy_to_boxes[toy]:
            if not visited[box]:
                visited[box] = True
                if box_to_toy[box] == -1 or dfs(box_to_toy[box], visited):
                    box_to_toy[box] = toy
                    return True
        return False
    
    max_matching = 0
    for toy in range(len(toy_sizes)):
        visited = [False] * len(box_sizes)
        if dfs(toy, visited):
            max_matching += 1
    
    return max_matching == len(toy_sizes)

print(solve())