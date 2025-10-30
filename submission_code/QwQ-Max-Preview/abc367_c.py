n, k = map(int, input().split())
r_list = list(map(int, input().split()))

result = []

def backtrack(current_seq, current_sum, index):
    if index == n:
        if current_sum % k == 0:
            result.append(current_seq.copy())
        return
    for num in range(1, r_list[index] + 1):
        current_seq.append(num)
        backtrack(current_seq, current_sum + num, index + 1)
        current_seq.pop()

backtrack([], 0, 0)

for seq in result:
    print(' '.join(map(str, seq)))