# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = set(map(int, data[2:]))
    
    total_sum = 0
    for i in range(1, K + 1):
        if i not in A:
            total_sum += i
    
    print(total_sum)

if __name__ == "__main__":
    main()