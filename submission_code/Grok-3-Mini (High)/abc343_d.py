import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
T = int(data[index])
index += 1
score = [0] * (N + 1)
freq = {0: N}
for _ in range(T):
    A = int(data[index])
    index += 1
    B = int(data[index])
    index += 1
    old_score = score[A]
    new_score = old_score + B
    freq[old_score] -= 1
    if freq[old_score] == 0:
        del freq[old_score]
    freq[new_score] = freq.get(new_score, 0) + 1
    score[A] = new_score
    print(len(freq))