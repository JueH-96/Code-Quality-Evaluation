import sys
input = sys.stdin.read

def count_valid_combinations(N, M, K, tests):
    from itertools import combinations

    # Generate all possible combinations of real keys
    all_combinations = list(combinations(range(1, N + 1), K))
    valid_combinations = set(all_combinations)

    for test in tests:
        keys, result = test[:-1], test[-1]
        real_keys_count = sum(1 for key in keys if key in valid_combinations)

        if result == 'o' and real_keys_count < K:
            return 0
        if result == 'x' and real_keys_count >= K:
            return 0

    return len(valid_combinations)

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    index = 3
    tests = []

    for _ in range(M):
        C = int(data[index])
        keys = list(map(int, data[index + 1:index + 1 + C]))
        result = data[index + 1 + C]
        tests.append((keys, result))
        index += 2 + C

    print(count_valid_combinations(N, M, K, tests))

if __name__ == "__main__":
    main()