def main():
    S = input().strip()
    allowed_strings = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}
    if S in allowed_strings:
        print("Yes")
    else:
        print("No")

main()