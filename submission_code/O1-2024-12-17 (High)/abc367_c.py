def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    R = list(map(int, data[2:]))

    sequence = []

    def backtrack(i, current_sum):
        if i == N:
            if current_sum % K == 0:
                print(" ".join(map(str, sequence)))
            return
        for val in range(1, R[i] + 1):
            sequence.append(val)
            backtrack(i + 1, current_sum + val)
            sequence.pop()

    backtrack(0, 0)

# Do not remove the next line
if __name__ == "__main__":
    main()