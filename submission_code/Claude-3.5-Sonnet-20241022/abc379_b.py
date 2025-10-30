# Read input
N, K = map(int, input().split())
S = input()

def max_strawberries(N, K, S):
    # Convert string to list for easier manipulation
    teeth = list(S)
    count = 0
    
    # Keep trying to find K consecutive healthy teeth
    while True:
        # Try to find K consecutive healthy teeth
        found = False
        for i in range(N - K + 1):
            # Check if we have K consecutive healthy teeth
            if all(teeth[j] == 'O' for j in range(i, i + K)):
                # Found K consecutive healthy teeth
                found = True
                # Mark these teeth as having cavities
                for j in range(i, i + K):
                    teeth[j] = 'X'
                count += 1
                break
        
        # If we couldn't find K consecutive healthy teeth, we're done
        if not found:
            break
    
    return count

# Print result
print(max_strawberries(N, K, S))