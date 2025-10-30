def main():
    # Read input
    n, x, y = map(int, input().split())
    dishes = []
    for _ in range(n):
        a, b = map(int, input().split())
        dishes.append((a, b))
    
    # Calculate the effective ratio for each dish
    # This represents how much of each limit (sweetness and saltiness) a dish consumes
    dishes_with_ratio = [(a, b, a/x + b/y) for a, b in dishes]
    
    # Sort dishes by effective ratio (from smallest to largest)
    # This prioritizes dishes that use the least proportion of our limits
    dishes_with_ratio.sort(key=lambda d: d[2])
    
    total_sweetness = 0
    total_saltiness = 0
    dishes_eaten = 0
    
    # Feed Snuke the dishes in order of effective ratio
    for a, b, _ in dishes_with_ratio:
        total_sweetness += a
        total_saltiness += b
        dishes_eaten += 1
        
        # If either limit is exceeded, Snuke stops eating
        if total_sweetness > x or total_saltiness > y:
            break
    
    print(dishes_eaten)

if __name__ == "__main__":
    main()