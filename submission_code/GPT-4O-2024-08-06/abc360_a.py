# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # Find the index of 'R' and 'M'
    index_R = S.index('R')
    index_M = S.index('M')
    
    # Check if 'R' is to the left of 'M'
    if index_R < index_M:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()