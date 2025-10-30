def generate_sequences(pos, curr_sum, sequence, N, K, R, result):
    if pos == N:
        if curr_sum % K == 0:
            result.append(sequence[:])
        return
        
    for i in range(1, R[pos] + 1):
        sequence.append(i)
        generate_sequences(pos + 1, curr_sum + i, sequence, N, K, R, result)
        sequence.pop()

N, K = map(int, input().split())
R = list(map(int, input().split()))

result = []
generate_sequences(0, 0, [], N, K, R, result)

for seq in sorted(result):
    print(*seq)