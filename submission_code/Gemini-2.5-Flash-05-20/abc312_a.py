import sys

def solve():
    S = sys.stdin.readline().strip()

    target_strings = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}

    if S in target_strings:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()