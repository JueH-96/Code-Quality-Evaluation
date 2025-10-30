def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    # Compute l[i]
    l = [0] * N
    for i in range(N):
        current_j = 0
        required = 1
        while i - current_j >= 0 and A[i - current_j] >= required:
            current_j += 1
            required += 1
        l[i] = current_j
    
    # Compute r[i] by reversing the array and using l_rev
    reversed_A = A[::-1]
    l_rev = [0] * N
    for i in range(N):
        current_j = 0
        required = 1
        while i - current_j >= 0 and reversed_A[i - current_j] >= required:
            current_j += 1
            required += 1
        l_rev[i] = current_j
    
    r = [0] * N
    for i in range(N):
        pos = N - 1 - i
        if pos < 0 or pos >= N:
            r[i] = 0
        else:
            r[i] = l_rev[pos]
    
    max_k = 0
    for i in range(N):
        possible_k = min(A[i], l[i] + 1, r[i] + 1)
        if possible_k > max_k:
            max_k = possible_k
    
    print(max_k)

if __name__ == '__main__':
    main()