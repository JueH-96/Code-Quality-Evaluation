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
    # This way, we try to minimize the number of dishes by picking the ones with the smallest sum first
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
    
    print(count)

if __name__ == "__main__":
    main()