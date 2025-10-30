from collections import Counter
from sys import stdin, setrecursionlimit

input = stdin.readline

setrecursionlimit(int(1e9))

def build(s):
	length = len(s)+1
	DP = [Counter() for _ in range(length)]
	DP[0][s[0]] = 1
	for i in range(1,length):
		DP[i] = DP[i-1] + Counter([s[i-1]])
	return DP

def main():
	input()
	line = input().strip()
	A = build(line)
	line = input().strip()
	B = build(line)
	
	if abs(len(A[-1])-len(B[-1])) > int(input()):
		print("No")
		return
	
	H = [0]*52
	for c,v in A[-1].items():
		H[ord(c)-97] = H[ord(c)-97] + v
	interset = 0
	
	for c,v in B[-1].items():
		a = ord(c)-97
		interset += min(v,H[a])

	if (len(A[-1]) + len(B[-1]) - 2*interset) > 2 * int(input()):
		print("No")
	else:
		print("Yes")
main()