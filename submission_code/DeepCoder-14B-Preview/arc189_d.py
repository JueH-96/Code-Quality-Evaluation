def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    result = []
    for K in range(1, N+1):
        idx = K - 1
        current_sum = A[idx]
        # Expand to the left
        left = idx - 1
        while left >= 0 and A[left] < current_sum:
            current_sum += A[left]
            left -= 1
        # Expand to the right
        right = idx + 1
        while right < N and A[right] < current_sum:
            current_sum += A[right]
            right += 1
        result.append(str(current_sum))
    print(' '.join(result))

if __name__ == '__main__':
    main()