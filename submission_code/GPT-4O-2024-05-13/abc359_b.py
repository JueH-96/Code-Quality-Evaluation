# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    count = 0
    
    for i in range(1, N+1):
        indices = [index for index, value in enumerate(A) if value == i]
        if indices[1] - indices[0] == 2:
            count += 1
    
    print(count)

if __name__ == "__main__":
    main()