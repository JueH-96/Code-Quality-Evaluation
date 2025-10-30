def can_transform(N, S, T):
    # Create a graph where edges represent character transformations
    graph = {}
    for c in 'abcdefghijklmnopqrstuvwxyz':
        graph[c] = set()
    
    # For each position, add edges between characters that need to be transformed
    for i in range(N):
        if S[i] != T[i]:
            graph[S[i]].add(T[i])
    
    # For each character in S, check if it needs to be transformed to multiple
    # different characters in T at different positions
    for c in S:
        if len(graph[c]) > 1:
            return -1
    
    # Create a set of characters that need to be transformed
    chars_to_transform = set()
    for i in range(N):
        if S[i] != T[i]:
            chars_to_transform.add(S[i])
    
    # If no transformations needed, return 0
    if not chars_to_transform:
        return 0
    
    # Check for cycles in transformations
    visited = set()
    def has_cycle(char, path):
        if char in path:
            return True
        if char in visited:
            return False
        visited.add(char)
        path.add(char)
        for next_char in graph[char]:
            if has_cycle(next_char, path):
                return True
        path.remove(char)
        return False
    
    # Check for cycles starting from each character that needs transformation
    for c in chars_to_transform:
        if has_cycle(c, set()):
            return -1
    
    # Count the number of transformations needed
    transformations = 0
    while chars_to_transform:
        # Find a leaf node (character that doesn't need to transform to anything)
        for c in chars_to_transform:
            is_leaf = True
            for target in graph[c]:
                if target in chars_to_transform:
                    is_leaf = False
                    break
            if is_leaf:
                transformations += 1
                chars_to_transform.remove(c)
                break
    
    return transformations

# Read input
N = int(input())
S = input()
T = input()

# Print result
print(can_transform(N, S, T))