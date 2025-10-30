import itertools

def main():
    N, K = map(int, input().split())
    functions = [tuple(map(int, input().split())) for _ in range(N)]
    
    max_value = float('-inf')
    for perm in itertools.permutations(range(N), K):
        value = 1
        for i in perm:
            A, B = functions[i]
            value = A * value + B
        max_value = max(max_value, value)
    
    print(max_value)

if __name__ == "__main__":
    main()