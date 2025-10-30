# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    total_sum = 0
    for i in range(N):
        for j in range(i + 1, N):
            total_sum += max(A[j] - A[i], 0)
    
    print(total_sum)

if __name__ == "__main__":
    main()