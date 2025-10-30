def get_all_possible_xors(N, stones):
    # Function to get all possible XOR values after moving stones between bags
    
    # Use set to store unique XOR values
    possible_xors = set()
    
    # Function to calculate XOR of all bags
    def calc_xor(bags):
        result = 0
        for x in bags:
            result ^= x
        return result
    
    # Try all possible combinations of moves using recursion
    def solve(bags, used):
        # Add current XOR to possible values
        possible_xors.add(calc_xor(bags))
        
        # Try all possible moves
        for i in range(N):
            if bags[i] == 0:  # Skip empty bags
                continue
            for j in range(N):
                if i == j:  # Can't move to same bag
                    continue
                    
                # Make move: add all stones from bag i to bag j
                new_bags = list(bags)
                new_bags[j] += new_bags[i]
                new_bags[i] = 0
                
                # Convert to tuple for hashing
                state = tuple(sorted(new_bags))
                if state not in used:
                    used.add(state)
                    solve(new_bags, used)
                    used.remove(state)
    
    # Start with initial state
    initial_bags = list(stones)
    initial_state = tuple(sorted(initial_bags))
    used = {initial_state}
    
    solve(initial_bags, used)
    return len(possible_xors)

# Read input
N = int(input())
A = list(map(int, input().split()))

# Get and print result
result = get_all_possible_xors(N, A)
print(result)