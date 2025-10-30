def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    A.sort()
    # Length of the sequence we keep (N-K)
    length = N - K
    
    answer = float('inf')
    
    # Slide over sorted array windows of size length, track min difference
    for i in range(N - length + 1):
        current_diff = A[i + length - 1] - A[i]
        if current_diff < answer:
            answer = current_diff

    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()