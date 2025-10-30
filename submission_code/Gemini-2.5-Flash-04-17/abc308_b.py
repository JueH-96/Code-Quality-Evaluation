# Read N and M
n, m = map(int, input().split())

# Read colors C (list of colors Takahashi ate)
c_colors = input().split()

# Read colors D (list of specific colors with non-default prices)
d_colors = input().split()

# Read prices P (P_0 followed by P_1 to P_M)
p_prices = list(map(int, input().split()))

# Extract the default price P_0
default_price = p_prices[0]

# Create a dictionary to map specific colors (D_i) to their prices (P_i)
# The input p_prices list has P_0 at index 0, P_1 at index 1, ..., P_M at index M.
# The input d_colors list has D_1 at index 0, D_2 at index 1, ..., D_M at index M-1.
# According to the problem description, D_i corresponds to P_i.
# So, d_colors[j] (representing D_{j+1}) corresponds to p_prices[j+1] (representing P_{j+1}) for j from 0 to M-1.
specific_prices = {}
for i in range(m):
    specific_prices[d_colors[i]] = p_prices[i + 1]

# Calculate the total cost
total_cost = 0
for color in c_colors:
    # Check if the plate color has a specific price defined in D
    if color in specific_prices:
        total_cost += specific_prices[color]
    else:
        # If not in D, use the default price P_0
        total_cost += default_price

# Print the total cost
print(total_cost)