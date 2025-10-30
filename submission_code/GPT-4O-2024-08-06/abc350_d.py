# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    from collections import defaultdict
    
    # Create a dictionary to store friendships
    friends = defaultdict(set)
    
    index = 2
    for _ in range(M):
        A = int(data[index])
        B = int(data[index + 1])
        friends[A].add(B)
        friends[B].add(A)
        index += 2
    
    # Set to store new friendships
    new_friendships = set()
    
    # Iterate over each user
    for X in range(1, N + 1):
        # Iterate over friends of X
        for Y in friends[X]:
            # Iterate over friends of Y
            for Z in friends[Y]:
                if Z != X and Z not in friends[X]:
                    # Add the new friendship in sorted order
                    new_friendships.add((min(X, Z), max(X, Z)))
    
    # Output the number of new friendships
    print(len(new_friendships))