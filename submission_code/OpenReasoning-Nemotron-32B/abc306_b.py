tokens = input().split()
tokens = tokens[:64]
if len(tokens) < 64:
	tokens.extend(['0'] * (64 - len(tokens)))
nums = [int(x) for x in tokens]
ans = 0
for i in range(64):
	ans += nums[i] * (1 << i)
print(ans)