# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # Since S is guaranteed to end with "2023", we can replace the last character with '4'
    modified_S = S[:-1] + '4'
    
    print(modified_S)

if __name__ == "__main__":
    main()