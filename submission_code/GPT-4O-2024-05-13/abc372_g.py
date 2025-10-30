# YOUR CODE HERE
import sys
input = sys.stdin.read

def count_valid_pairs(N, A, B, C):
    min_x = float('inf')
    min_y = float('inf')
    for i in range(N):
        min_x = min(min_x, (C[i] - 1) // A[i])
        min_y = min(min_y, (C[i] - 1) // B[i])
    
    count = 0
    for x in range(1, min_x + 1):
        for y in range(1, min_y + 1):
            valid = True
            for i in range(N):
                if A[i] * x + B[i] * y >= C[i]:
                    valid = False
                    break
            if valid:
                count += 1
    return count

def main():
    data = input().split()
    index = 0
    T = int(data[index])
    index += 1
    results = []
    for _ in range(T):
        N = int(data[index])
        index += 1
        A = []
        B = []
        C = []
        for _ in range(N):
            A.append(int(data[index]))
            B.append(int(data[index + 1]))
            C.append(int(data[index + 2]))
            index += 3
        results.append(count_valid_pairs(N, A, B, C))
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()