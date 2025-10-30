from itertools import product

def find_sequences(N, K, R):
    for seq in product(*(range(1, r + 1) for r in R)):
        if sum(seq) % K == 0:
            yield seq

def main():
    N, K = map(int, input().split())
    R = list(map(int, input().split()))
    for seq in find_sequences(N, K, R):
        print(*seq)

if __name__ == "__main__":
    main()