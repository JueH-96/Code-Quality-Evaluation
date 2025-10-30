def main():
    # Read the floating point value of temperature
    X = float(input().strip())
    
    # Determine the classification based on temperature value
    if X >= 38.0:
        print(1)
    elif X >= 37.5:
        print(2)
    else:
        print(3)

# Call the main function to execute the program
if __name__ == '__main__':
    main()