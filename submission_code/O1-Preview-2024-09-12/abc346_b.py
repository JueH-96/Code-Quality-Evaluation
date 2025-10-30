# YOUR CODE HERE
W,B = map(int, input().split())

pattern = 'wbwbwwbwbwbw'
pattern_length = len(pattern)

N = W+B+pattern_length

S = (pattern * ((N // pattern_length)+2))[:N]

w_prefix = [0]*(N+1)
b_prefix = [0]*(N+1)

for i in range(N):
    w_prefix[i+1] = w_prefix[i] + (1 if S[i]=='w' else 0)
    b_prefix[i+1] = b_prefix[i] + (1 if S[i]=='b' else 0)

counts_set = set()

for l in range(N):
    # Maximum substring length is W+B (up to 200)
    max_r = min(N-1, l+W+B-1)
    for r in range(l, max_r+1):
        w_count = w_prefix[r+1] - w_prefix[l]
        b_count = b_prefix[r+1] - b_prefix[l]
        counts_set.add((w_count, b_count))

if (W,B) in counts_set:
    print("Yes")
else:
    print("No")