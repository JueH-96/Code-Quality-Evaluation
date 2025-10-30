# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    D = int(data[index])
    index += 1
    
    snakes = []
    for _ in range(N):
        T = int(data[index])
        index += 1
        L = int(data[index])
        index += 1
        snakes.append((T, L))
    
    results = []
    for k in range(1, D + 1):
        max_weight = 0
        for T, L in snakes:
            weight = T * (L + k)
            if weight > max_weight:
                max_weight = weight
        results.append(max_weight)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()