import sys
import itertools

def calculate_max_value(N, K, functions):
    max_value = float('-inf')
    for p in itertools.permutations(range(N), K):
        value = 1
        for i in p:
            value = functions[i][0] * value + functions[i][1]
        max_value = max(max_value, value)
    return max_value

def main():
    N, K = map(int, sys.stdin.readline().split())
    functions = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    max_value = calculate_max_value(N, K, functions)
    print(max_value)

if __name__ == "__main__":
    main()