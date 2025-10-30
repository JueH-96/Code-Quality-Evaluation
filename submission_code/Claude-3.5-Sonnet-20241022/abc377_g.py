def min_cost_to_match(target, prev_strings):
    # If we want to make it empty, cost is length of target
    min_cost = len(target)
    
    # Try to match with each previous string
    for prev in prev_strings:
        cost = 0
        t_len = len(target)
        p_len = len(prev)
        
        # Find the longest common prefix
        common = 0
        while common < min(t_len, p_len) and target[common] == prev[common]:
            common += 1
            
        # Cost = delete remaining chars in target + add remaining chars from prev
        cost = (t_len - common) + (p_len - common)
        min_cost = min(min_cost, cost)
        
    return min_cost

def main():
    # Read input
    N = int(input())
    strings = []
    for _ in range(N):
        strings.append(input().strip())
    
    # Process each string
    for k in range(N):
        # For each string k, consider all previous strings as potential matches
        prev_strings = strings[:k]
        target = strings[k]
        
        # Calculate minimum cost
        result = min_cost_to_match(target, prev_strings)
        print(result)

if __name__ == "__main__":
    main()