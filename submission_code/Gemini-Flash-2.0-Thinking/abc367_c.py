def solve():
    n, k = map(int, input().split())
    r = list(map(int, input().split()))

    def generate(index, current_sequence):
        if index == n:
            if sum(current_sequence) % k == 0:
                print(*current_sequence)
            return

        for value in range(1, r[index] + 1):
            current_sequence.append(value)
            generate(index + 1, current_sequence)
            current_sequence.pop()

    generate(0, [])

solve()