import sys
from functools import reduce

def read_input():
    N, K = map(int, sys.stdin.readline().split())
    functions = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    return N, K, functions

def solve(N, K, functions):
    sequence = [int(sys.stdin.readline()) for _ in range(K)]
    return max(reduce(lambda x, f: f[0]*x + f[1], [functions[i-1] for i in sequence], 1) for sequence in permutations(range(1, N+1), K))

def main():
    N, K, functions = read_input()
    print(solve(N, K, functions))

if __name__ == "__main__":
    main()