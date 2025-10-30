def main():
    R = int(input().strip())
    if R >= 1 and R <= 99:
        print(100 - R)
    elif R >= 100 and R <= 199:
        print(200 - R)
    elif R >= 200 and R <= 299:
        print(300 - R)

if __name__ == "__main__":
    main()