# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    W = int(data[0])
    H = int(data[1])
    N = int(data[2])
    strawberries = [(int(data[3 + 2 * i]), int(data[4 + 2 * i])) for i in range(N)]
    A = int(data[3 + 2 * N])
    a = [0] + [int(x) for x in data[4 + 2 * N:4 + 2 * N + A]] + [W]
    B = int(data[4 + 2 * N + A])
    b = [0] + [int(y) for y in data[5 + 2 * N + A:5 + 2 * N + A + B]] + [H]
    
    min_strawberries = float('inf')
    max_strawberries = float('-inf')
    
    for i in range(A + 1):
        for j in range(B + 1):
            count = 0
            for (x, y) in strawberries:
                if a[i] < x < a[i + 1] and b[j] < y < b[j + 1]:
                    count += 1
            min_strawberries = min(min_strawberries, count)
            max_strawberries = max(max_strawberries, count)
    
    print(min_strawberries, max_strawberries)

if __name__ == "__main__":
    main()