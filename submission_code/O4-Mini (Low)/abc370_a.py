def main():
    L, R = map(int, input().split())
    # If exactly one hand is raised
    if L + R == 1:
        # If left hand is raised, he wants to eat takoyaki
        if L == 1:
            print("Yes")
        # Otherwise only right hand is raised, he does not want to eat
        else:
            print("No")
    else:
        # Either both hands or no hands are raised
        print("Invalid")

if __name__ == "__main__":
    main()