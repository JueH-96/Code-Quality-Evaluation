import sys

def solve():
    s = sys.stdin.readline().strip()
    
    valid_strings = ["ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"]
    
    if s in valid_strings:
        print("Yes")
    else:
        print("No")

solve()