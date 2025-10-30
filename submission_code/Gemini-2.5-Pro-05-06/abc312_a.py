# YOUR CODE HERE
def solve():
    s = input()
    
    target_strings = ["ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"]
    
    if s in target_strings:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()