def find_palindrome(N):
    # Check if N can be represented as a single valid number
    s = str(N)
    if '0' not in s:
        return s
    
    # Use BFS to generate all possible factor sequences
    from collections import deque
    visited = set()
    queue = deque()
    queue.append((N, []))
    
    while queue:
        current_n, current_seq = queue.popleft()
        
        # Try all possible factors
        for i in range(1, current_n + 1):
            if current_n % i == 0:
                i_str = str(i)
                if '0' in i_str:
                    continue  # Skip factors with '0'
                
                new_seq = current_seq + [i_str]
                new_n = current_n // i
                
                # Check if the new sequence forms a palindrome when combined with remaining factors
                # If new_n is 1, we can form a complete sequence
                if new_n == 1:
                    s_candidate = '*'.join(new_seq)
                    if s_candidate == s_candidate[::-1]:
                        return s_candidate
                else:
                    # Add the remaining part as a factor if it's valid
                    remaining = new_n
                    remaining_str = str(remaining)
                    if '0' not in remaining_str:
                        full_seq = new_seq + [remaining_str]
                        s_candidate = '*'.join(full_seq)
                        if s_candidate == s_candidate[::-1]:
                            return s_candidate
                    # Enqueue the new state for further factorization
                    state = (new_n, new_seq)
                    if state not in visited:
                        visited.add(state)
                        queue.append(state)
    
    # If no valid sequence found
    return -1

# Read input
N = int(input())
# Find the palindrome string
result = find_palindrome(N)
# Output the result
print(result if result != -1 else -1)