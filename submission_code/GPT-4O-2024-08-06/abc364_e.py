# YOUR CODE HERE
def max_dishes(N, X, Y, dishes):
    # Sort dishes by the sum of sweetness and saltiness
    dishes.sort(key=lambda dish: dish[0] + dish[1])
    
    total_sweetness = 0
    total_saltiness = 0
    count = 0
    
    for sweetness, saltiness in dishes:
        if total_sweetness + sweetness <= X and total_saltiness + saltiness <= Y:
            total_sweetness += sweetness
            total_saltiness += saltiness
            count += 1
        else:
            break
    
    return count

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    
    dishes = []
    index = 3
    for _ in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        dishes.append((A, B))
        index += 2
    
    result = max_dishes(N, X, Y, dishes)
    print(result)