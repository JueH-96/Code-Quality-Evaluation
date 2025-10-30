def main():
    import sys
    
    numbers = []
    for line in sys.stdin:
        val = int(line.strip())
        numbers.append(val)
        if val == 0:
            break
    
    for val in reversed(numbers):
        print(val)

# Do not remove or rename this call
main()