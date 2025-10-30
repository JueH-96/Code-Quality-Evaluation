def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    C = list(map(int, data[2:]))
    
    # We need to find the minimum cost to make exactly one pair of adjacent characters the same.
    # We will consider each pair (i, i+1) and calculate the cost to make them the same.
    
    min_cost = float('inf')
    
    # Iterate over each possible pair (i, i+1)
    for i in range(N - 1):
        if S[i] == S[i + 1]:
            # They are already the same, no cost needed to make them the same
            current_cost = 0
        else:
            # They are different, we need to change either S[i] or S[i+1]
            # We should change the one with the smaller cost
            current_cost = min(C[i], C[i + 1])
        
        # Check if this is the minimum cost found so far
        if current_cost < min_cost:
            min_cost = current_cost
    
    print(min_cost)

if __name__ == "__main__":
    main()