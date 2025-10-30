def main():
    allowed = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}
    S = input().strip()
    if S in allowed:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()