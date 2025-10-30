n = int(input().strip())
dice = []
for _ in range(n):
    data = list(map(int, input().split()))
    k = data[0]
    faces = data[1:]
    freq = {}
    for num in faces:
        freq[num] = freq.get(num, 0) + 1
    dice.append((k, freq))

max_prob = 0.0
n_dice = len(dice)
for i in range(n_dice):
    k_i, freq_i = dice[i]
    for j in range(i + 1, n_dice):
        k_j, freq_j = dice[j]
        total_common = 0
        if len(freq_i) <= len(freq_j):
            for num, cnt_i in freq_i.items():
                cnt_j = freq_j.get(num, 0)
                if cnt_j > 0:
                    total_common += cnt_i * cnt_j
        else:
            for num, cnt_j in freq_j.items():
                cnt_i = freq_i.get(num, 0)
                if cnt_i > 0:
                    total_common += cnt_i * cnt_j
        prob = total_common / (k_i * k_j)
        if prob > max_prob:
            max_prob = prob

print("{:.15f}".format(max_prob))