def max_total_price(T, test_cases):
    results = []
    
    for case in test_cases:
        N, K, cakes = case
        prices = []
        
        # Calculate the price for each possible pair of cakes
        for i in range(N):
            for j in range(i + 1, N):
                price = max(cakes[i][0] + cakes[j][0], cakes[i][1] + cakes[j][1], cakes[i][2] + cakes[j][2])
                prices.append(price)
        
        # Sort prices in descending order and take the top K
        prices.sort(reverse=True)
        total_price = sum(prices[:K])
        results.append(total_price)
    
    return results

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

T = int(data[0])
test_cases = []
index = 1

for _ in range(T):
    N, K = map(int, data[index].split())
    cakes = []
    for i in range(N):
        X, Y, Z = map(int, data[index + 1 + i].split())
        cakes.append((X, Y, Z))
    test_cases.append((N, K, cakes))
    index += N + 1

# Get results
results = max_total_price(T, test_cases)

# Print results
for result in results:
    print(result)