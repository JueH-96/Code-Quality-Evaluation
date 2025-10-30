S = input().strip()
count = [0] * 26
sum_indices = [0] * 26
total = 0

for idx, c in enumerate(S):
    char_idx = ord(c) - ord('A')
    current_count = count[char_idx]
    current_sum = sum_indices[char_idx]
    contribution = current_count * (idx - 1) - current_sum
    total += contribution
    count[char_idx] += 1
    sum_indices[char_idx] += idx

print(total)