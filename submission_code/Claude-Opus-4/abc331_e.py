# YOUR CODE HERE
N, M, L = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Store forbidden pairs in a set for O(1) lookup
forbidden = set()
for _ in range(L):
    c, d = map(int, input().split())
    forbidden.add((c-1, d-1))  # Convert to 0-indexed

# Create lists of (price, index) and sort by price descending
main_dishes = [(a[i], i) for i in range(N)]
side_dishes = [(b[i], i) for i in range(M)]
main_dishes.sort(reverse=True)
side_dishes.sort(reverse=True)

max_price = 0

# Try combinations starting from most expensive
# We can stop early once we find a valid combination that's guaranteed to be optimal
for main_price, main_idx in main_dishes:
    for side_price, side_idx in side_dishes:
        current_price = main_price + side_price
        
        # If current price is less than max found, we can stop checking this main dish
        if current_price <= max_price:
            break
            
        # Check if this combination is allowed
        if (main_idx, side_idx) not in forbidden:
            max_price = current_price
            break

print(max_price)