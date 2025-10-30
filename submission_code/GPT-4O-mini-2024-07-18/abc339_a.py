# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # Split the string by '.' and take the last part
    last_substring = S.split('.')[-1]
    
    print(last_substring)

if __name__ == "__main__":
    main()