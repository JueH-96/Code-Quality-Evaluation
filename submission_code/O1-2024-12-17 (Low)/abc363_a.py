def main():
    R = int(input().strip())
    
    # Based on the rating R, determine the current bracket:
    #  1–99   => 1 caret
    # 100–199 => 2 carets
    # 200–299 => 3 carets
    #
    # The goal is to find the smallest increase in R such that
    # the rating moves to the next bracket (thus displaying one more caret).
    
    if R <= 99:
        # Currently 1 caret, next bracket starts at 100
        print(100 - R)
    elif R <= 199:
        # Currently 2 carets, next bracket starts at 200
        print(200 - R)
    else:
        # Currently 3 carets, next bracket starts at 300
        print(300 - R)

# Do not forget to call main()
if __name__ == "__main__":
    main()