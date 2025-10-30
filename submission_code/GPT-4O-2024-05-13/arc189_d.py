# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    left_max = [0] * N
    right_max = [0] * N
    
    # Calculate left_max
    for i in range(N):
        if i == 0:
            left_max[i] = A[i]
        else:
            if A[i] > left_max[i-1]:
                left_max[i] = A[i]
            else:
                left_max[i] = left_max[i-1] + A[i]
    
    # Calculate right_max
    for i in range(N-1, -1, -1):
        if i == N-1:
            right_max[i] = A[i]
        else:
            if A[i] > right_max[i+1]:
                right_max[i] = A[i]
            else:
                right_max[i] = right_max[i+1] + A[i]
    
    # Calculate the result for each slime
    result = [0] * N
    for i in range(N):
        result[i] = max(left_max[i], right_max[i])
    
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()