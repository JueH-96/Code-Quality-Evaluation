N, M, L = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Store forbidden combinations
forbidden = set()
for _ in range(L):
    c, d = map(int, input().split())
    forbidden.add((c-1, d-1))  # Convert to 0-indexed

# Create sorted indices by price (descending)
main_indices = sorted(range(N), key=lambda i: a[i], reverse=True)
side_indices = sorted(range(M), key=lambda i: b[i], reverse=True)

max_price = 0

# Try combinations starting from most expensive main dishes
for main_idx in main_indices:
    found_valid = False
    for side_idx in side_indices:
        if (main_idx, side_idx) not in forbidden:
            max_price = max(max_price, a[main_idx] + b[side_idx])
            found_valid = True
            break
    
    # If we found a valid combination for this main dish,
    # and no future main dish can give us a better result, we can stop
    if found_valid and a[main_idx] + b[side_indices[0]] <= max_price:
        break

print(max_price)