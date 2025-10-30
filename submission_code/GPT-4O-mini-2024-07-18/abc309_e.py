def main():
    import sys
    from collections import defaultdict, deque

    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and M
    N, M = map(int, data[0].split())
    
    # Read parent relationships
    parents = list(map(int, data[1].split()))
    
    # Build the tree structure
    tree = defaultdict(list)
    for i in range(2, N + 1):
        tree[parents[i - 2]].append(i)
    
    # Read insurance information
    insurances = []
    for i in range(2, 2 + M):
        x, y = map(int, data[i].split())
        insurances.append((x, y))
    
    # Set to track covered people
    covered = set()
    
    # Function to perform BFS to find all descendants
    def cover_descendants(person, generations):
        queue = deque([(person, 0)])  # (current_person, current_generation)
        while queue:
            current, gen = queue.popleft()
            if current in covered:
                continue
            covered.add(current)
            if gen < generations:
                for child in tree[current]:
                    queue.append((child, gen + 1))
    
    # Process each insurance
    for x, y in insurances:
        cover_descendants(x, y)
    
    # Output the number of unique covered people
    print(len(covered))

if __name__ == "__main__":
    main()