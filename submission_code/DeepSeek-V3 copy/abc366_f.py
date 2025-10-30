import sys
import itertools

def main():
    N, K = map(int, sys.stdin.readline().split())
    functions = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    
    max_value = -1
    
    # Since K is up to 10, we can afford to generate all permutations
    for p in itertools.permutations(range(N), K):
        current = 1
        for idx in p:
            A, B = functions[idx]
            current = A * current + B
        if current > max_value:
            max_value = current
    
    print(max_value)

if __name__ == "__main__":
    main()