import sys

def main():
    S = sys.stdin.readline().strip()
    targets = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}
    print("Yes" if S in targets else "No")

if __name__ == "__main__":
    main()