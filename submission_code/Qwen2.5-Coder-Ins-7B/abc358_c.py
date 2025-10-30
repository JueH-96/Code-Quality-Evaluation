import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
S = data[2:]

def min_stands_to_visit(N, M, S):
    # Create a list to keep track of which stands sell each flavor
    stands_by_flavor = [[] for _ in range(M)]
    
    # Populate the list with the indices of stands that sell each flavor
    for i in range(N):
        for j in range(M):
            if S[i][j] == 'o':
                stands_by_flavor[j].append(i)
    
    # Use a greedy algorithm to select the minimum number of stands
    visited_stands = set()
    for i in range(M):
        if i not in visited_stands:
            # Select the stand that sells the current flavor and all subsequent flavors
            selected_stand = stands_by_flavor[i][0]
            for j in range(i, M):
                if selected_stand not in stands_by_flavor[j]:
                    selected_stand = stands_by_flavor[j][0]
            visited_stands.add(selected_stand)
    
    return len(visited_stands)

# Call the function and print the result
print(min_stands_to_visit(N, M, S))