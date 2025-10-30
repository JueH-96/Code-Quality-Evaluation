def main():
    L, R = map(int, input().split())
    
    # If Snuke is raising both or none -> "Invalid"
    if (L == 1 and R == 1) or (L == 0 and R == 0):
        print("Invalid")
    else:
        # If Snuke is raising left -> "Yes"
        if L == 1:
            print("Yes")
        # If Snuke is raising right -> "No"
        else:
            print("No")

if __name__ == "__main__":
    main()