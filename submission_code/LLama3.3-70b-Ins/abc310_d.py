import sys
import itertools

def read_input():
    N, T, M = map(int, sys.stdin.readline().split())
    incompatible_pairs = []
    for _ in range(M):
        A, B = map(int, sys.stdin.readline().split())
        incompatible_pairs.append((A-1, B-1))
    return N, T, M, incompatible_pairs

def is_valid_division(division, incompatible_pairs):
    for pair in incompatible_pairs:
        if division[pair[0]] == division[pair[1]]:
            return False
    return True

def count_divisions(N, T, M, incompatible_pairs):
    count = 0
    for division in itertools.product(range(T), repeat=N):
        if len(set(division)) == T and is_valid_division(division, incompatible_pairs):
            count += 1
    return count

def main():
    N, T, M, incompatible_pairs = read_input()
    result = count_divisions(N, T, M, incompatible_pairs)
    print(result)

if __name__ == "__main__":
    main()