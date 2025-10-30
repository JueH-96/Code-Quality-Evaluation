# YOUR CODE HERE
def main():
    s = input().strip()
    
    # Determine the majority character by inspecting the first three letters
    if s[0] == s[1]:
        majority = s[0]
    else:
        if s[0] == s[2]:
            majority = s[0]
        else:
            majority = s[1]
    
    # Find and print the 1-based index of the differing character
    for i, ch in enumerate(s):
        if ch != majority:
            print(i + 1)
            return

# Don't forget to call main!
main()