def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1
    K_list = []
    for _ in range(Q):
        K_list.append(int(input[idx]))
        idx += 1

    possible = set()
    for a in range(N + 1):
        for b in range(N + 1):
            current = a * N + b * N - a * b
            possible.add(current)

    for k in K_list:
        if k in possible:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()