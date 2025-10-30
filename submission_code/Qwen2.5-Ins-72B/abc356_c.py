# YOUR CODE HERE
import itertools

def solve(N, M, K, tests):
    def is_valid_combination(combination, tests):
        for C, A, R in tests:
            real_keys = sum(combination[a - 1] for a in A)
            if (R == 'o' and real_keys < K) or (R == 'x' and real_keys >= K):
                return False
        return True

    valid_combinations = 0
    for combination in itertools.product([0, 1], repeat=N):
        if is_valid_combination(combination, tests):
            valid_combinations += 1

    return valid_combinations

if __name__ == "__main__":
    N, M, K = map(int, input().split())
    tests = []
    for _ in range(M):
        line = input().split()
        C = int(line[0])
        A = list(map(int, line[1:-1]))
        R = line[-1]
        tests.append((C, A, R))

    print(solve(N, M, K, tests))