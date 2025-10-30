# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    
    B, G = map(int, data.split())
    
    if B > G:
        print("Bat")
    else:
        print("Glove")

if __name__ == "__main__":
    main()