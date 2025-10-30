def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:N+1]))
    H = [0] + H  # 1-based indexing

    stack = []
    k = [0] * (N + 2)  # k[1..N]
    for j in range(1, N + 1):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        if not stack:
            k[j] = 0
        else:
            k[j] = stack[-1]
        stack.append(j)

    diff = [0] * (N + 2)
    for j in range(1, N + 1):
        a = k[j] + 1
        b = j - 1
        if a <= b:
            diff[a] += 1
            diff[b + 1] -= 1
        else:
            if j - 1 >= 1:
                diff[j - 1] += 1
                diff[j] -= 1

    # Compute prefix sum to get the result
    current = 0
    c = [0] * (N + 1)
    for i in range(1, N + 1):
        current += diff[i]
        c[i] = current

    print(' '.join(map(str, c[1:N+1])))

if __name__ == '__main__':
    main()