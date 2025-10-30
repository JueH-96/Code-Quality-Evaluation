from math import factorial

def solve():
    n, k = map(int, input().split())
    nk = n * k

    memo = {}
    def count_sequences(counts):
        key = tuple(sorted(counts.items()))
        if key in memo:
            return memo[key]

        total_sum = sum(counts.values())
        if total_sum == 0:
            return 1

        numerator = factorial(total_sum)
        denominator = 1
        for count in counts.values():
            denominator *= factorial(count)

        result = numerator // denominator
        memo[key] = result
        return result

    initial_counts = {i: k for i in range(1, n + 1)}
    s = count_sequences(initial_counts)
    target_index = (s + 1) // 2

    result = []
    current_counts = initial_counts.copy()

    for _ in range(nk):
        for num in range(1, n + 1):
            if current_counts[num] > 0:
                current_counts_copy = current_counts.copy()
                current_counts_copy[num] -= 1
                remaining_sequences = count_sequences(current_counts_copy)

                if target_index <= remaining_sequences:
                    result.append(num)
                    current_counts = current_counts_copy
                    break
                else:
                    target_index -= remaining_sequences
        else:
            raise Exception("Error in logic")

    print(*result)

solve()