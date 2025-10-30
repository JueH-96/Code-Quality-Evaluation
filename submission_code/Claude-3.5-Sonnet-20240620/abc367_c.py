# YOUR CODE HERE
def generate_sequences(N, K, R):
    def backtrack(index, current_sum, sequence):
        if index == N:
            if current_sum % K == 0:
                print(*sequence)
            return

        for i in range(1, R[index] + 1):
            sequence.append(i)
            backtrack(index + 1, current_sum + i, sequence)
            sequence.pop()

    N, K = map(int, input().split())
    R = list(map(int, input().split()))

    backtrack(0, 0, [])