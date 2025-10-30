def main():
    import sys
    input = sys.stdin.readline

    # Read number of snakes N and maximum increase D
    N, D = map(int, input().strip().split())
    
    # Read snake characteristics: thickness and length for each snake
    snakes = [tuple(map(int, input().strip().split())) for _ in range(N)]
    
    # For each increment k from 1 to D, compute the max weight (thickness * (length+k))
    for k in range(1, D + 1):
        max_weight = 0
        for T, L in snakes:
            weight = T * (L + k)
            if weight > max_weight:
                max_weight = weight
        print(max_weight)

if __name__ == '__main__':
    main()