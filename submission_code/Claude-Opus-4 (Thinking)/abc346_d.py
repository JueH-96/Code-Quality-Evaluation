# YOUR CODE HERE
n = int(input())
s = input().strip()
costs = list(map(int, input().split()))

min_cost = float('inf')

# Try each position where adjacent characters are the same
for i in range(n - 1):
    # Try both '0' and '1' as the repeated character
    for same_char in '01':
        # Build the target string
        target = [''] * n
        target[i] = target[i + 1] = same_char
        
        # Fill to the left with alternating pattern
        for j in range(i - 1, -1, -1):
            target[j] = '0' if target[j + 1] == '1' else '1'
        
        # Fill to the right with alternating pattern
        for j in range(i + 2, n):
            target[j] = '0' if target[j - 1] == '1' else '1'
        
        # Calculate the cost of transforming s to target
        cost = sum(costs[j] for j in range(n) if s[j] != target[j])
        
        min_cost = min(min_cost, cost)

print(min_cost)