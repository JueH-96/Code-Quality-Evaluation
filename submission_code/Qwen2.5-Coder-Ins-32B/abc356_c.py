# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    M = int(input[index + 1])
    K = int(input[index + 2])
    index += 3

    tests = []
    for _ in range(M):
        C = int(input[index])
        keys = list(map(int, input[index + 1:index + 1 + C]))
        result = input[index + 1 + C]
        tests.append((keys, result))
        index += 1 + C + 1

    valid_combinations = 0

    for combination in range(1 << N):
        real_keys = [i + 1 for i in range(N) if (combination & (1 << i)) != 0]
        valid = True
        for keys, result in tests:
            real_count = sum(1 for key in keys if key in real_keys)
            if result == 'o' and real_count < K:
                valid = False
                break
            if result == 'x' and real_count >= K:
                valid = False
                break
        if valid:
            valid_combinations += 1

    print(valid_combinations)

if __name__ == "__main__":
    main()