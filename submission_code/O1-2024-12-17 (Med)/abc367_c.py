def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    R = list(map(int, data[2:]))

    def dfs(i, current_seq, current_sum):
        if i == N:
            if current_sum % K == 0:
                print(' '.join(map(str, current_seq)))
            return

        for val in range(1, R[i] + 1):
            dfs(i + 1, current_seq + [val], current_sum + val)

    dfs(0, [], 0)

# Do not forget to call main
if __name__ == "__main__":
    main()