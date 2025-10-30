# YOUR CODE HERE
import sys

def main():
    R = int(sys.stdin.read().strip())
    
    if 1 <= R <= 99:
        print(100 - R)
    elif 100 <= R <= 199:
        print(200 - R)
    elif 200 <= R <= 299:
        print(300 - R)

if __name__ == "__main__":
    main()