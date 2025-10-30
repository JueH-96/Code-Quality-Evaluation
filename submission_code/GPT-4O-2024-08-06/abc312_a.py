# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    valid_strings = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}
    
    if S in valid_strings:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()