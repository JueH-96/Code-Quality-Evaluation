# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    
    A = list(map(int, data[index:index + N]))
    index += N
    
    B = list(map(int, data[index:index + N]))
    index += N
    
    K = int(data[index])
    index += 1
    
    results = []
    
    for _ in range(K):
        X_k = int(data[index])
        Y_k = int(data[index + 1])
        index += 2
        
        total_sum = 0
        for i in range(X_k):
            for j in range(Y_k):
                total_sum += abs(A[i] - B[j])
        
        results.append(total_sum)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()