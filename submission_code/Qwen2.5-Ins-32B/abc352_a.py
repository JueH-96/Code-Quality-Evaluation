# YOUR CODE HERE
def check_station_stop(N, X, Y, Z):
    # Check if the travel is inbound or outbound
    if X < Y:
        # Inbound travel
        return Z > X and Z < Y
    else:
        # Outbound travel
        return Z < X and Z > Y

# Read input from stdin
N, X, Y, Z = map(int, input().split())

# Determine if the train stops at station Z
result = check_station_stop(N, X, Y, Z)

# Write the answer to stdout
print("Yes" if result else "No")