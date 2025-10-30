def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # positions[i] will store the (1-based) indices where color i appears
    positions = [[] for _ in range(N+1)]
    for idx, color in enumerate(A):
        positions[color].append(idx + 1)

    answer = 0
    for color in range(1, N+1):
        p1, p2 = positions[color]
        if abs(p2 - p1) == 2:
            answer += 1

    print(answer)

# Do not forget to call main()
if __name__ == '__main__':
    main()