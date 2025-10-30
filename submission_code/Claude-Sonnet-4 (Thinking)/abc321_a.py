N = input().strip()

# Check if digits are strictly decreasing
is_321_like = all(int(N[i]) > int(N[i + 1]) for i in range(len(N) - 1))

print("Yes" if is_321_like else "No")