def main():
    R = int(input().strip())
    
    if R <= 99:
        # Next bracket starts at 100
        print(100 - R)
    elif R <= 199:
        # Next bracket starts at 200
        print(200 - R)
    else:
        # Next bracket starts at 300
        print(300 - R)

# Call main() to execute
main()