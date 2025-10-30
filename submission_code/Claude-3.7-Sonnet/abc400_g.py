def solve_case(n, k, cakes):
    # Calculate the price of each possible pair
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            cake_i = cakes[i]
            cake_j = cakes[j]
            price = max(cake_i[0] + cake_j[0], cake_i[1] + cake_j[1], cake_i[2] + cake_j[2])
            pairs.append((price, i, j))
    
    # Sort pairs by price (descending)
    pairs.sort(reverse=True)
    
    # Greedy selection of top K pairs
    total_price = 0
    used_cakes = set()
    selected_pairs = 0
    
    for price, i, j in pairs:
        if i not in used_cakes and j not in used_cakes:
            total_price += price
            used_cakes.add(i)
            used_cakes.add(j)
            selected_pairs += 1
            
            if selected_pairs == k:
                break
    
    return total_price

def main():
    t = int(input())
    
    for _ in range(t):
        n, k = map(int, input().split())
        cakes = []
        
        for _ in range(n):
            x, y, z = map(int, input().split())
            cakes.append((x, y, z))
        
        print(solve_case(n, k, cakes))

if __name__ == "__main__":
    main()