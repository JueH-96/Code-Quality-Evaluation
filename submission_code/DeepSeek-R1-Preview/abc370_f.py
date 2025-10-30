def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    A = list(map(int, input[2:2+N]))
    total = sum(A)
    
    # Binary search for x
    left = 1
    right = total // K
    x = 0
    while left <= right:
        mid = (left + right) // 2
        # Check if possible to split into K groups with each sum >= mid
        B = A + A
        n = len(B)
        P = [0] * (n + 1)
        for i in range(n):
            P[i+1] = P[i] + B[i]
        next_jump = [0] * n
        j = 0
        current_sum = 0
        for i in range(n):
            while j < n and current_sum < mid:
                current_sum += B[j]
                j += 1
            if current_sum >= mid:
                next_jump[i] = j
            else:
                next_jump[i] = -1
            if i < j:
                current_sum -= B[i]
        possible = False
        for s in range(N):
            current = s
            count = 0
            while count <= K and current < s + N:
                if next_jump[current] == -1:
                    break
                next_pos = next_jump[current]
                if next_pos > s + N:
                    break
                current = next_pos
                count += 1
            if current >= s + N and count <= K:
                possible = True
                break
        if possible:
            x = mid
            left = mid + 1
        else:
            right = mid - 1
    
    # Compute y
    y = 0
    for i in range(N):
        current = A[i]
        next_val = A[(i+1) % N]
        if current + next_val < x:
            y += 1
    
    print(x, y)

if __name__ == '__main__':
    main()