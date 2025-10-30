def main():
    import sys

    N, M = map(int, sys.stdin.readline().split())
    has_taro = [False] * (N + 1)

    for _ in range(M):
        parts = sys.stdin.readline().strip().split()
        if len(parts) != 2:
            A_i = int(parts[0])
            B_i = 'F'  # Default to 'F' if B_i is missing
        else:
            A_i, B_i = parts
            A_i = int(A_i)
        if B_i == 'M' and not has_taro[A_i]:
            print("Yes")
            has_taro[A_i] = True
        else:
            print("No")

if __name__ == "__main__":
    main()