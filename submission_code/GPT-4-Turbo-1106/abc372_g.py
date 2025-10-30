from math import gcd

def count_pairs(A, B, C):
    # Find the maximum x and y that satisfy the condition for each i
    x_max = min(C[i] // A[i] for i in range(len(A)))
    y_max = min(C[i] // B[i] for i in range(len(B)))

    # Count the number of valid pairs (x, y)
    count = 0
    for x in range(1, x_max + 1):
        for y in range(1, y_max + 1):
            if all(A[i] * x + B[i] * y < C[i] for i in range(len(A))):
                count += 1
    return count

def main():
    T = int(input().strip())
    results = []
    for _ in range(T):
        N = int(input().strip())
        A, B, C = [], [], []
        for _ in range(N):
            a, b, c = map(int, input().strip().split())
            A.append(a)
            B.append(b)
            C.append(c)
        results.append(count_pairs(A, B, C))
    for result in results:
        print(result)

if __name__ == "__main__":
    main()