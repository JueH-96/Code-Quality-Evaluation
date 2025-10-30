def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = []
    B = []
    idx = 1
    for _ in range(N):
        a = int(input_data[idx]); b = int(input_data[idx+1])
        A.append(a); B.append(b)
        idx += 2

    sum_A = sum(A)
    max_diff = max(b - a for a, b in zip(A, B))

    print(sum_A + max_diff)

def main():
    solve()

if __name__ == "__main__":
    main()