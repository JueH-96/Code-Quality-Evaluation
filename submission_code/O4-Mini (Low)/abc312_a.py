def main():
    import sys
    S = sys.stdin.readline().strip()
    valid = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}
    print("Yes" if S in valid else "No")

if __name__ == "__main__":
    main()