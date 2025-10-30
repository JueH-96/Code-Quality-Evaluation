import itertools

def can_eat_all(perm, X, Y, A, B):
    total_sweetness = 0
    total_saltiness = 0
    for idx in perm:
        total_sweetness += A[idx]
        total_saltiness += B[idx]
        if total_sweetness > X or total_saltiness > Y:
            return False
    return True

def max_dishes_eaten(N, X, Y, A, B):
    for length in range(N, 0, -1):
        for perm in itertools.permutations(range(N), length):
            if can_eat_all(perm, X, Y, A, B):
                return length
    return 0

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    A = []
    B = []

    for i in range(N):
        A.append(int(data[3 + 2*i]))
        B.append(int(data[4 + 2*i]))

    result = max_dishes_eaten(N, X, Y, A, B)
    print(result)

if __name__ == "__main__":
    main()