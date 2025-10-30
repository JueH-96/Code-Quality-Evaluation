# YOUR CODE HERE
import sys
import heapq

def solve():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        K = int(data[index + 1])
        index += 2
        
        cakes = []
        for i in range(N):
            X = int(data[index])
            Y = int(data[index + 1])
            Z = int(data[index + 2])
            cakes.append((X, Y, Z))
            index += 3
        
        # Generate all possible pairs and their prices
        pair_prices = []
        
        for i in range(N):
            for j in range(i + 1, N):
                X1, Y1, Z1 = cakes[i]
                X2, Y2, Z2 = cakes[j]
                price = max(X1 + X2, Y1 + Y2, Z1 + Z2)
                pair_prices.append((price, i, j))
        
        # Sort pair prices in descending order
        pair_prices.sort(reverse=True, key=lambda x: x[0])
        
        # Select the top K pairs
        used = [False] * N
        total_price = 0
        pairs_selected = 0
        
        for price, i, j in pair_prices:
            if not used[i] and not used[j]:
                total_price += price
                used[i] = True
                used[j] = True
                pairs_selected += 1
                if pairs_selected == K:
                    break
        
        results.append(total_price)
    
    for result in results:
        print(result)