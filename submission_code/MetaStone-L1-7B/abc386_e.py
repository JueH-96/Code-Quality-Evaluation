import itertools

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    
    max_xor = 0
    for comb in itertools.combinations(A, K):
        current = 0
        for num in comb:
            current ^= num
        if current > max_xor:
            max_xor = current
    print(max_xor)

if __name__ == '__main__':
    main()