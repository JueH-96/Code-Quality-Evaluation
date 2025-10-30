# YOUR CODE HERE
def main():
    import sys

    import sys

    def input():
        return sys.stdin.read()

    data = input().split()
    if len(data) < 2:
        N = int(data[0])
        S = ''
        while len(S) < N:
            S += sys.stdin.readline().strip()
    else:
        N = int(data[0])
        S = ''.join(data[1:])

    left_ones = [0] * (N + 1)
    for i in range(1, N + 1):
        if S[i - 1] == '1':
            left_ones[i] = left_ones[i - 1] + 1
        else:
            left_ones[i] = 0

    right_twos = [0] * (N + 1)
    for i in range(N - 1, -1, -1):
        if i + 1 < N and S[i + 1] == '2':
            right_twos[i] = right_twos[i + 1] + 1
        else:
            right_twos[i] = 0

    max_length = 1  # At least one '/' exists
    for i in range(N):
        if S[i] == '/':
            k = min(left_ones[i], right_twos[i])
            current_length = 2 * k + 1
            if current_length > max_length:
                max_length = current_length

    print(max_length)

if __name__ == "__main__":
    main()