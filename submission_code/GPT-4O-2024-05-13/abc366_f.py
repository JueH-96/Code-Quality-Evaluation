# YOUR CODE HERE
import itertools

def compute_function_value(A, B, x):
    return A * x + B

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    
    A = []
    B = []
    
    index = 2
    for i in range(N):
        A.append(int(data[index]))
        B.append(int(data[index + 1]))
        index += 2
    
    max_value = float('-inf')
    
    for perm in itertools.permutations(range(N), K):
        value = 1
        for i in perm:
            value = compute_function_value(A[i], B[i], value)
        max_value = max(max_value, value)
    
    print(max_value)

if __name__ == "__main__":
    main()