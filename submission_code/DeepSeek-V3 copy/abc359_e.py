def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    
    A = [0] * (N + 1)
    operations = [0] * N
    total_ops = 0
    
    for i in range(N):
        # The number of operations needed for A[i] to become positive is (H[i] + 1) * (i + 1)
        # Because each step increases A[i] by 1, and it takes (H[i] + 1) steps to make A[i] > 0
        # But considering the previous steps, it's more complex
        # After analyzing, the formula is: operations[i] = operations[i-1] + H[i] + 1
        if i == 0:
            operations[i] = H[i] + 1
        else:
            operations[i] = operations[i-1] + H[i] + 1
    
    print(' '.join(map(str, operations)))

if __name__ == "__main__":
    main()