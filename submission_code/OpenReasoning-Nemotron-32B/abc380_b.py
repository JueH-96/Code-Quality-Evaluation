def main():
	s = input().strip()
	tokens = s.split('|')
	segments = tokens[1:-1]
	result = [str(len(segment)) for segment in segments]
	print(" ".join(result))

if __name__ == "__main__":
	main()