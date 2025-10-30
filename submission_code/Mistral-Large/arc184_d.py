import sys
from itertools import combinations

MOD = 998244353

def count_remaining_sets(N, balls):
    # Generate all possible subsets of balls
    all_subsets = []
    for r in range(1, N + 1):
        all_subsets.extend(combinations(range(N), r))

    # Function to check if a subset is valid
    def is_valid_subset(subset):
        for k in subset:
            remaining = set(subset)
            remaining.remove(k)
            for i in remaining:
                if (balls[i][0] < balls[k][0] and balls[i][1] < balls[k][1]) or (balls[i][0] > balls[k][0] and balls[i][1] > balls[k][1]):
                    return False
        return True

    # Count the number of valid subsets
    valid_count = 0
    for subset in all_subsets:
        if is_valid_subset(subset):
            valid_count = (valid_count + 1) % MOD

    return valid_count

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    balls = []
    index = 1
    for i in range(N):
        X_i = int(data[index])
        Y_i = int(data[index + 1])
        balls.append((X_i, Y_i))
        index += 2

    result = count_remaining_sets(N, balls)
    print(result)

if __name__ == "__main__":
    main()