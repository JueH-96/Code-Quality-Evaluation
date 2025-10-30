def main():
    import sys
    S = sys.stdin.readline().strip()
    valid = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}
    if S in valid:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()