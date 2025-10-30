def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    
    A = []
    B = []
    index = 2
    for _ in range(N):
        A.append(int(data[index]))
        B.append(int(data[index + 1]))
        index += 2
    
    used = [False] * N
    a = 1
    b = 0
    
    for _ in range(K):
        max_val = -float('inf')
        best_i = -1
        best_a = 0
        best_b = 0
        for i in range(N):
            if used[i]:
                continue
            ai = A[i]
            bi = B[i]
            new_a = a * ai
            new_b = b + a * bi
            current_val = new_a + new_b
            if current_val > max_val:
                max_val = current_val
                best_i = i
                best_a = new_a
                best_b = new_b
        if best_i == -1:
            break
        used[best_i] = True
        a = best_a
        b = best_b
    
    print(a + b)

if __name__ == '__main__':
    main()