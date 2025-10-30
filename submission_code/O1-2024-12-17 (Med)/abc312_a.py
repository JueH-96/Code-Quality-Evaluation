def main():
    S = input().strip()
    valid_strings = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}
    if S in valid_strings:
        print("Yes")
    else:
        print("No")

main()