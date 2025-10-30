# YOUR CODE HERE
n = int(input())
pairs = []
for _ in range(n):
    l, r = map(int, input().split())
    pairs.append((l, r))

# Calculate the range of possible sums
min_sum = sum(l for l, r in pairs)
max_sum = sum(r for l, r in pairs)

# Check if 0 is achievable
if min_sum > 0 or max_sum < 0:
    print("No")
else:
    print("Yes")
    
    # Start with all minimum values
    result = [l for l, r in pairs]
    current_sum = min_sum
    
    # Adjust values to make sum = 0
    target_increase = 0 - current_sum
    
    for i in range(n):
        if target_increase <= 0:
            break
        
        l, r = pairs[i]
        # How much can we increase this element?
        max_increase = r - l
        # How much should we increase it?
        increase = min(target_increase, max_increase)
        
        result[i] += increase
        target_increase -= increase
    
    print(*result)