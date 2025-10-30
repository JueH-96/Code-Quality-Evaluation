# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    r = list(map(int, input().split()))

    ans = []

    def generate_sequences(current_sequence):
        if len(current_sequence) == n:
            if sum(current_sequence) % k == 0:
                ans.append(current_sequence)
            return

        for i in range(1, r[len(current_sequence)] + 1):
            generate_sequences(current_sequence + [i])

    generate_sequences([])

    for seq in ans:
        print(*seq)

solve()