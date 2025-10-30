def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    from collections import defaultdict
    
    friends = defaultdict(set)
    
    index = 2
    for _ in range(M):
        A = int(data[index])
        B = int(data[index + 1])
        friends[A].add(B)
        friends[B].add(A)
        index += 2
    
    new_friendships = 0
    
    # Iterate over each user
    for user in range(1, N + 1):
        # To avoid counting duplicates and self-loops
        visited = set()
        visited.add(user)
        
        # Check friends of friends
        for friend in list(friends[user]):
            for fof in list(friends[friend]):
                if fof not in visited and fof not in friends[user]:
                    new_friendships += 1
                    friends[user].add(fof)
                    friends[fof].add(user)
                visited.add(fof)
    
    print(new_friendships)

if __name__ == "__main__":
    main()