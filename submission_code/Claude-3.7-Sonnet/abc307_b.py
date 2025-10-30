def is_palindrome(s):
    return s == s[::-1]

def main():
    # Read the number of strings
    n = int(input())
    
    # Read all the strings
    strings = [input() for _ in range(n)]
    
    # Check all pairs of strings
    for i in range(n):
        for j in range(n):
            # Make sure i and j are distinct
            if i != j:
                # Check if the concatenation is a palindrome
                if is_palindrome(strings[i] + strings[j]):
                    print("Yes")
                    return
    
    print("No")

if __name__ == "__main__":
    main()