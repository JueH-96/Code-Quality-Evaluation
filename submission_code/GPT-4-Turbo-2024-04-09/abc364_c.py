def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+2*N]))
    
    # We need to find the minimum number of dishes he will end up eating
    # such that the total sweetness or total saltiness exceeds the given limits.
    
    # Sort dishes by a heuristic that maximizes the impact on both sweetness and saltiness
    # to potentially minimize the number of dishes eaten.
    dishes = sorted(zip(A, B), key=lambda x: -(x[0] + x[1]))
    
    total_sweetness = 0
    total_saltiness = 0
    count = 0
    
    for a, b in dishes:
        total_sweetness += a
        total_saltiness += b
        count += 1
        if total_sweetness > X or total_saltiness > Y:
            print(count)
            return

main()