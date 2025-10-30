# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+2*N]))
    
    # Combine A and B into a list of tuples
    dishes = list(zip(A, B))
    
    # Sort the dishes based on the sum of A_i and B_i in ascending order
    # This is a heuristic to try to minimize the number of dishes
    dishes.sort(key=lambda x: x[0] + x[1])
    
    total_A = 0
    total_B = 0
    count = 0
    
    for a, b in dishes:
        total_A += a
        total_B += b
        count += 1
        if total_A > X or total_B > Y:
            break
    
    # Now, try sorting in different ways to find the minimum count
    # Sort by A in ascending order
    dishes.sort(key=lambda x: x[0])
    
    total_A = 0
    total_B = 0
    count_A = 0
    
    for a, b in dishes:
        total_A += a
        total_B += b
        count_A += 1
        if total_A > X or total_B > Y:
            break
    
    # Sort by B in ascending order
    dishes.sort(key=lambda x: x[1])
    
    total_A = 0
    total_B = 0
    count_B = 0
    
    for a, b in dishes:
        total_A += a
        total_B += b
        count_B += 1
        if total_A > X or total_B > Y:
            break
    
    # Sort by A in descending order
    dishes.sort(key=lambda x: -x[0])
    
    total_A = 0
    total_B = 0
    count_A_desc = 0
    
    for a, b in dishes:
        total_A += a
        total_B += b
        count_A_desc += 1
        if total_A > X or total_B > Y:
            break
    
    # Sort by B in descending order
    dishes.sort(key=lambda x: -x[1])
    
    total_A = 0
    total_B = 0
    count_B_desc = 0
    
    for a, b in dishes:
        total_A += a
        total_B += b
        count_B_desc += 1
        if total_A > X or total_B > Y:
            break
    
    # Find the minimum count among all sorting strategies
    min_count = min(count, count_A, count_B, count_A_desc, count_B_desc)
    
    print(min_count)

if __name__ == "__main__":
    main()