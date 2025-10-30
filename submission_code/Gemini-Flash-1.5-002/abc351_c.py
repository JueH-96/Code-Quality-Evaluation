# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

seq = []
for a in A:
    seq.append(2**a)
    while len(seq) >= 2 and seq[-1] == seq[-2]:
        seq[-2] = seq[-1] + seq[-2]
        seq.pop()
print(len(seq))