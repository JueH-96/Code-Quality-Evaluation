# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    R = int(input().strip())
    
    if 1 <= R <= 99:
        # To increase from 1 ^ to 2 ^, we need to reach 100
        print(100 - R)
    elif 100 <= R <= 199:
        # To increase from 2 ^ to 3 ^, we need to reach 200
        print(200 - R)
    elif 200 <= R <= 299:
        # To increase from 3 ^ to 4 ^, we need to reach 300
        print(300 - R)

if __name__ == "__main__":
    main()