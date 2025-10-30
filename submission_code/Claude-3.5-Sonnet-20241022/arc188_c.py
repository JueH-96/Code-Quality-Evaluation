def check_valid(N, M, testimonies, confused):
    # Try all possible honest/liar assignments
    for mask in range(1 << N):
        valid = True
        # Check each testimony
        for a, b, c in testimonies:
            a -= 1  # Convert to 0-based indexing
            b -= 1
            
            # Determine if A is honest after considering confusion
            a_honest = ((mask >> a) & 1) == 1
            if confused[a]:
                a_honest = not a_honest
                
            # Determine if B is honest
            b_honest = ((mask >> b) & 1) == 1
            
            # What A says about B
            if a_honest:
                # Honest (not confused) or Liar (confused) tells truth
                if c == 0 and not b_honest:  # Says honest but B is liar
                    valid = False
                    break
                if c == 1 and b_honest:  # Says liar but B is honest
                    valid = False
                    break
            else:
                # Liar (not confused) or Honest (confused) tells lies
                if c == 0 and b_honest:  # Says honest but B is honest
                    valid = False
                    break
                if c == 1 and not b_honest:  # Says liar but B is liar
                    valid = False
                    break
        
        if valid:
            return True
    return False

def solve():
    # Read input
    N, M = map(int, input().split())
    testimonies = []
    for _ in range(M):
        a, b, c = map(int, input().split())
        testimonies.append((a, b, c))
    
    # Try all possible combinations of confused villagers
    confused = [False] * N
    
    # Special case: if M = 0, return all zeros
    if M == 0:
        print("0" * N)
        return
    
    # Try each possible combination of confused villagers
    for mask in range(1 << N):
        for i in range(N):
            confused[i] = ((mask >> i) & 1) == 1
            
        if check_valid(N, M, testimonies, confused):
            # Found a valid solution
            print("".join("1" if c else "0" for c in confused))
            return
    
    # No solution found
    print("-1")

# Run the solution
solve()