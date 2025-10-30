# Read input values
H, W, N = map(int, input().split())
holed_squares = set(tuple(map(int, input().split())) for _ in range(N))

# Initialize the count of holeless squares
holeless_count = 0

# Check each possible top-left corner of a square region
for i in range(1, H + 1):
    for j in range(1, W + 1):
        # Check each possible size of the square region
        for n in range(1, min(H - i + 2, W - j + 2)):
            # Check if the square region is holeless
            is_holeless = True
            for k in range(n):
                for l in range(n):
                    if (i + k, j + l) in holed_squares:
                        is_holeless = False
                        break
                if not is_holeless:
                    break
            if is_holeless:
                holeless_count += 1
            else:
                break

# Print the number of holeless squares
print(holeless_count)