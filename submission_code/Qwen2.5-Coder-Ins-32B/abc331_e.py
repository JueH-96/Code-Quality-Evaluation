# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    
    N = int(input[index])
    M = int(input[index + 1])
    L = int(input[index + 2])
    index += 3
    
    a = list(map(int, input[index:index + N]))
    index += N
    
    b = list(map(int, input[index:index + M]))
    index += M
    
    forbidden = set()
    for _ in range(L):
        c = int(input[index]) - 1
        d = int(input[index + 1]) - 1
        forbidden.add((c, d))
        index += 2
    
    a.sort(reverse=True)
    b.sort(reverse=True)
    
    max_price = 0
    for i in range(N):
        for j in range(M):
            if (i, j) not in forbidden:
                max_price = max(max_price, a[i] + b[j])
                break  # No need to check further side dishes for this main dish
    
    print(max_price)

if __name__ == "__main__":
    main()