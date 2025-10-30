def main():
    import sys
    
    numbers = []
    
    for line in sys.stdin:
        num = int(line.strip())
        numbers.append(num)
        if num == 0:
            break
    
    for num in reversed(numbers):
        print(num)

# Call main function
main()