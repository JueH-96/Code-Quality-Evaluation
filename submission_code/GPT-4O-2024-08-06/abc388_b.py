# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = int(data[1])
    
    snakes = []
    index = 2
    for _ in range(N):
        T = int(data[index])
        L = int(data[index + 1])
        snakes.append((T, L))
        index += 2
    
    for k in range(1, D + 1):
        max_weight = 0
        for T, L in snakes:
            weight = T * (L + k)
            if weight > max_weight:
                max_weight = weight
        print(max_weight)

if __name__ == "__main__":
    main()