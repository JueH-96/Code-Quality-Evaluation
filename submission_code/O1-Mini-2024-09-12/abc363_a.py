# YOUR CODE HERE
def main():
    R = int(input())
    if 1 <= R <= 99:
        L = 100
    elif 100 <= R <= 199:
        L = 200
    elif 200 <= R <= 299:
        L = 300
    else:
        # According to constraints, R <= 299
        L = R
    delta = L - R
    print(delta)

if __name__ == "__main__":
    main()