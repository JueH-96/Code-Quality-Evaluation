# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    total_sum = 0
    
    for i in range(N-1):
        for j in range(i+1, N):
            max_val = max(A[i], A[j])
            min_val = min(A[i], A[j])
            total_sum += max_val // min_val
    
    print(total_sum)

if __name__ == "__main__":
    main()