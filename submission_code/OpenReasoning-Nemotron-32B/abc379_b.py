def main():
	n, k = map(int, input().split())
	s = input().strip()
	segments = s.split('X')
	ans = 0
	for seg in segments:
		ans += len(seg) // k
	print(ans)

if __name__ == '__main__':
	main()