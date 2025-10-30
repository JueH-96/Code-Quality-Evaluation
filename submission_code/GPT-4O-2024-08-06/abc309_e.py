# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    # Read parent list
    parents = [0] * (N + 1)
    for i in range(2, N + 1):
        parents[i] = int(data[index])
        index += 1
    
    # Read insurance data
    insurances = []
    for _ in range(M):
        x = int(data[index])
        index += 1
        y = int(data[index])
        index += 1
        insurances.append((x, y))
    
    # Build the tree
    from collections import defaultdict, deque
    
    children = defaultdict(list)
    for i in range(2, N + 1):
        children[parents[i]].append(i)
    
    # Set to track covered people
    covered = set()
    
    # Process each insurance
    for x, y in insurances:
        # BFS to find all descendants up to y generations
        queue = deque([(x, 0)])  # (current person, current generation)
        while queue:
            current, generation = queue.popleft()
            if generation > y:
                continue
            if current not in covered:
                covered.add(current)
                for child in children[current]:
                    queue.append((child, generation + 1))
    
    # Output the number of unique covered people
    print(len(covered))