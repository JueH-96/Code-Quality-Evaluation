import math
from itertools import permutations

def factorial(n):
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n+1):
        result *= i
    return result

def count_sequences(N, K):
    total = factorial(N * K)
    for i in range(1, N+1):
        total = total // factorial(K)
    return total

def get_sequence(N, K, target):
    sequence = []
    counts = [K] * N
    for _ in range(N * K):
        for i in range(N):
            if counts[i] == 0:
                continue
            # Calculate the number of sequences if we choose i+1 at this position
            temp_counts = counts.copy()
            temp_counts[i] -= 1
            # Calculate the factorial of the remaining counts
            remaining = factorial(sum(temp_counts))
            for cnt in temp_counts:
                remaining = remaining // factorial(cnt)
            if remaining >= target:
                sequence.append(i+1)
                counts[i] -= 1
                break
            else:
                target -= remaining
    return sequence

def main():
    N, K = map(int, input().split())
    total_sequences = count_sequences(N, K)
    target = (total_sequences + 1) // 2
    sequence = get_sequence(N, K, target)
    print(' '.join(map(str, sequence)))

if __name__ == "__main__":
    main()