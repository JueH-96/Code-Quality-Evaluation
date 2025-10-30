def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = []
    B = []
    idx = 1
    for _ in range(N):
        a = int(input_data[idx]); b = int(input_data[idx+1])
        idx += 2
        A.append(a)
        B.append(b)

    sumA = sum(A)
    max_diff = max(b - a for a, b in zip(A, B))
    print(sumA + max_diff)

# Do not forget to call main()
if __name__ == "__main__":
    main()