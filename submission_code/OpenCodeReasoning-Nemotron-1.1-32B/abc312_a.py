def main():
	s = input().strip()
	valid_set = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}
	if s in valid_set:
		print("Yes")
	else:
		print("No")

if __name__ == "__main__":
	main()