import sys
from itertools import combinations

def count_valid_combinations(N, M, K, tests):
    valid_combinations = 0
    for real_keys in combinations(range(1, N+1), K):
        valid = True
        for C, keys, result in tests:
            if (result == 'o' and len(set(keys) & set(real_keys)) < K) or (result == 'x' and len(set(keys) & set(real_keys)) >= K):
                valid = False
                break
        if valid:
            valid_combinations += 1
    return valid_combinations

def main():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N, M, K = int(data[index]), int(data[index+1]), int(data[index+2])
    index += 3
    
    tests = []
    for _ in range(M):
        C = int(data[index])
        keys = list(map(int, data[index+1:index+C+1]))
        result = data[index+C+1]
        tests.append((C, keys, result))
        index += C + 2
    
    print(count_valid_combinations(N, M, K, tests))

if __name__ == "__main__":
    main()