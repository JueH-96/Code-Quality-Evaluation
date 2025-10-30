def solve():
    import sys
    
    numbers = []
    for line in sys.stdin:
        value = int(line.strip())
        numbers.append(value)
        if value == 0:
            break
    
    # Reverse the list and print
    for num in reversed(numbers):
        print(num)

def main():
    solve()

if __name__ == "__main__":
    main()