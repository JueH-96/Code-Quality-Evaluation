# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    index = 1
    A = [[0] * (i + 1) for i in range(N)]
    
    for i in range(N):
        for j in range(i + 1):
            A[i][j] = int(data[index])
            index += 1
    
    current_element = 1
    for i in range(1, N + 1):
        if current_element <= i:
            current_element = A[i-1][current_element-1]
        else:
            current_element = A[current_element-1][i-1]
    
    print(current_element)

if __name__ == "__main__":
    main()