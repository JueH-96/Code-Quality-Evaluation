def main():
	valid_strings = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}
	s = input().strip()
	if s in valid_strings:
		print("Yes")
	else:
		print("No")

if __name__ == "__main__":
	main()