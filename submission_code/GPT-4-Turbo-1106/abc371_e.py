from collections import defaultdict

def read_input():
    N = int(input().strip())
    A = list(map(int, input().strip().split()))
    return N, A

def calculate_subsequence_sum(N, A):
    last_seen = defaultdict(int)
    sum_f = 0
    current_sum = 0

    for j in range(1, N + 1):
        if last_seen[A[j - 1]] == 0:
            current_sum += 1
        else:
            current_sum -= last_seen[A[j - 1]] - 1
        last_seen[A[j - 1]] = j
        sum_f += current_sum

    return sum_f

def main():
    N, A = read_input()
    result = calculate_subsequence_sum(N, A)
    print(result)

if __name__ == "__main__":
    main()