from itertools import combinations

def count_ways(N, T, M, incompatible_pairs):
    # Create a graph representation of incompatible pairs
    incompatible_graph = [[] for _ in range(N + 1)]
    for a, b in incompatible_pairs:
        incompatible_graph[a].append(b)
        incompatible_graph[b].append(a)

    # Function to check if a team assignment is valid
    def is_valid_assignment(assignment):
        for a in range(1, N + 1):
            for b in incompatible_graph[a]:
                if assignment[a] == assignment[b]:
                    return False
        return True

    # Count valid assignments using backtracking
    def backtrack(player, assignment):
        if player > N:
            if is_valid_assignment(assignment):
                return 1
            return 0
        
        total_ways = 0
        for team in range(1, T + 1):
            assignment[player] = team
            total_ways += backtrack(player + 1, assignment)
            assignment[player] = 0  # Reset assignment
        
        return total_ways

    # Start backtracking from player 1
    assignment = [0] * (N + 1)  # 0 means unassigned
    return backtrack(1, assignment)

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N, T, M = map(int, data[0].split())
    incompatible_pairs = [tuple(map(int, line.split())) for line in data[1:M + 1]]
    
    result = count_ways(N, T, M, incompatible_pairs)
    print(result)

if __name__ == "__main__":
    main()