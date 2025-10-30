import sys

N = int(input())
S = list(input())

takahashi_wins = 0
aoki_wins = 0

for i, res in enumerate(S):
	if res == "T":
		takahashi_wins += 1
	else:
		aoki_wins += 1

	if takahashi_wins > aoki_wins + N - i - 1:
		print("T")
		sys.exit()
	if aoki_wins > takahashi_wins + N - i - 1:
		print("A")
		sys.exit()

if takahashi_wins > aoki_wins:
	print("T")
elif aoki_wins > takahashi_wins:
	print("A")
else:
	for i, res in enumerate(S):
		if res == "T":
			print("T")
			sys.exit()

	print("A")