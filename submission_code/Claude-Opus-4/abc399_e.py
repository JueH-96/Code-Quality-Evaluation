# YOUR CODE HERE
def solve():
    n = int(input())
    s = input().strip()
    t = input().strip()
    
    # Create a mapping from each character in S to what it needs to become in T
    # Also check if it's possible
    char_map = {}
    for i in range(n):
        if s[i] in char_map:
            if char_map[s[i]] != t[i]:
                print(-1)
                return
        else:
            char_map[s[i]] = t[i]
    
    # Build a directed graph where edge (a, b) means we need to transform a to b
    # Count the number of transformations needed
    operations = 0
    for src, dst in char_map.items():
        if src != dst:
            operations += 1
    
    print(operations)

solve()