def main():
    R = int(input().strip())
    
    if 1 <= R <= 99:
        # Next category starts at 100
        print(100 - R)
    elif 100 <= R <= 199:
        # Next category starts at 200
        print(200 - R)
    else:  # 200 <= R <= 299
        # Next category starts at 300
        print(300 - R)

# Do not remove or modify this call
if __name__ == "__main__":
    main()