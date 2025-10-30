# YOUR CODE HERE
def count_valid_divisions(N, T, M, incompatible_pairs):
    # Convert incompatible pairs into a set for fast lookup
    incompatible_set = set(incompatible_pairs)
    
    # This will store the number of valid divisions
    valid_divisions_count = 0
    
    # Helper function to check if a current assignment is valid
    def is_valid_assignment(assignment):
        for a, b in incompatible_set:
            if assignment[a - 1] == assignment[b - 1]:
                return False
        return True
    
    # Recursive function to try all possible assignments
    def assign_players(player_index, assignment):
        nonlocal valid_divisions_count
        
        if player_index == N:
            # If all players are assigned, check if it's a valid assignment
            if is_valid_assignment(assignment):
                valid_divisions_count += 1
            return
        
        # Try assigning current player to each team from 0 to T-1
        for team in range(T):
            assignment[player_index] = team
            assign_players(player_index + 1, assignment)
            assignment[player_index] = -1  # Backtrack
    
    # Start with an empty assignment
    initial_assignment = [-1] * N
    assign_players(0, initial_assignment)
    
    return valid_divisions_count

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    M = int(data[2])
    
    incompatible_pairs = []
    index = 3
    for _ in range(M):
        A = int(data[index])
        B = int(data[index + 1])
        incompatible_pairs.append((A, B))
        index += 2
    
    result = count_valid_divisions(N, T, M, incompatible_pairs)
    print(result)

main()