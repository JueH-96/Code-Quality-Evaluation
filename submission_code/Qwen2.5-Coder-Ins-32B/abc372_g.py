# YOUR CODE HERE
import sys

def count_valid_pairs(N, A, B, C):
    max_x = max((C[i] - 1) // A[i] for i in range(N) if A[i] > 0)
    max_y = max((C[i] - 1) // B[i] for i in range(N) if B[i] > 0)
    
    count = 0
    for x in range(1, max_x + 1):
        for y in range(1, max_y + 1):
            if all(A[i] * x + B[i] * y < C[i] for i in range(N)):
                count += 1
    return count

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    T = int(input[index])
    index += 1
    results = []
    for _ in range(T):
        N = int(input[index])
        index += 1
        A = []
        B = []
        C = []
        for _ in range(N):
            A.append(int(input[index]))
            B.append(int(input[index + 1]))
            C.append(int(input[index + 2]))
            index += 3
        results.append(count_valid_pairs(N, A, B, C))
    for result in results:
        print(result)

if __name__ == "__main__":
    main()