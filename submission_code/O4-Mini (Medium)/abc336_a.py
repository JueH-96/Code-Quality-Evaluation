def main():
    N = int(input().strip())
    # Construct and print the Dragon String of level N: 'L' + N times 'o' + 'ng'
    print("L" + "o" * N + "ng")

# Call main to execute the program
main()